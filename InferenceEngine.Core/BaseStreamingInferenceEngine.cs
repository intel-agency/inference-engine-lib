using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace InferenceEngine.Core;

/// <summary>
/// Reuses <see cref="BaseInferenceEngine{TInput, TOutput}"/> for session/acceleration/warmup/
/// model-download/diagnostics, and adds an autoregressive decode loop driven by engine-supplied
/// hooks. One-shot consumers see no change: <see cref="PredictAsync"/> is routed through the
/// streaming loop and reduced via <see cref="BuildFinalOutput"/>.
/// </summary>
/// <typeparam name="TInput">Raw engine input (e.g. a prompt).</typeparam>
/// <typeparam name="TOutput">Fully-reduced one-shot result.</typeparam>
/// <typeparam name="TStep">One streamed model-level step.</typeparam>
public abstract class BaseStreamingInferenceEngine<TInput, TOutput, TStep>
    : BaseInferenceEngine<TInput, TOutput>, IStreamingInferenceEngine<TInput, TOutput, TStep>
{
    // Single-in-flight-stream guard (§6.9): KV-cache state is per-run; a second concurrent
    // stream on the same instance must be rejected.
    private int _activeStreams;

    /// <summary>Creates the streaming engine, forwarding options + logger to the base.</summary>
    protected BaseStreamingInferenceEngine(InferenceEngineOptions options, ILogger? logger = null)
        : base(options, logger)
    {
    }

    /// <summary>
    /// The <see cref="StreamingOptions"/> for the stream currently in progress, or <see langword="null"/>
    /// outside a stream. Hooks (<see cref="BuildStepInputs"/>/<see cref="ProcessStepOutputs"/>) read this
    /// to honor <see cref="StreamingOptions.IncludePrefillInStream"/>, <see cref="StreamingOptions.EosTokenIds"/>, etc.
    /// </summary>
    protected StreamingOptions? CurrentOptions { get; private set; }

    /// <summary>Default streaming options used when a caller passes <see langword="null"/>.</summary>
    protected virtual StreamingOptions DefaultStreamingOptions => new();

    /// <summary>One-shot behavior is preserved: run the stream to completion and reduce it.</summary>
    public override async Task<TOutput> PredictAsync(TInput input)
    {
        var steps = new List<TStep>();
        await foreach (var step in PredictStreamingAsync(input).ConfigureAwait(false))
            steps.Add(step);
        return BuildFinalOutput(steps);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<TStep> PredictStreamingAsync(
        TInput input,
        StreamingOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await LoadAsync().ConfigureAwait(false);

        if (Interlocked.CompareExchange(ref _activeStreams, 1, 0) != 0)
            throw new InvalidOperationException(
                "Only one in-flight stream is supported per engine instance (§6.9).");

        var opts = options ?? DefaultStreamingOptions;
        CurrentOptions = opts;
        object? state = null;

        try
        {
            for (int step = 0; step < opts.MaxSteps; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var (inputs, buildContext) = BuildStepInputs(input, state, step);

                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs;
                try
                {
                    outputs = RunStep(inputs);
                }
                finally
                {
                    DisposeNamedOnnxValues(inputs);
                }

                TStep yielded;
                object? nextState;
                bool isEos;
                try
                {
                    (yielded, nextState, isEos) = ProcessStepOutputs(outputs, buildContext);
                }
                finally
                {
                    outputs.Dispose();
                }

                // Guard against an engine that reuses (mutates in place) the same state object
                // across steps: disposing the prior reference would also dispose the live state.
                if (state is IDisposable priorState && !ReferenceEquals(state, nextState))
                    priorState.Dispose();
                state = nextState;
                yield return yielded;

                if (isEos) yield break;
            }
        }
        finally
        {
            // Reset per-run state and release the in-flight guard on every exit path
            // (success, EOS, MaxSteps, exception, and cancellation).
            if (state is IDisposable finalState)
                finalState.Dispose();
            state = null;
            CurrentOptions = null;
            Interlocked.Exchange(ref _activeStreams, 0);
        }
    }

    // ---- Model-aware hooks: the engine knows its own I/O contract. ----------------

    /// <summary>
    /// Build the ONNX inputs for a given step and return the carry-forward context.
    /// <paramref name="step"/> == 0 means prefill (full prompt); step &gt; 0 means decode
    /// (feed the previous step's <c>present_key_values</c> back as <c>past_key_values</c>).
    /// </summary>
    protected abstract (List<NamedOnnxValue> Inputs, object? BuildContext) BuildStepInputs(
        TInput input, object? state, int step);

    /// <summary>
    /// Parse one forward pass's outputs into a streamed step, the next carry-forward state, and an
    /// end-of-sequence signal. The engine is responsible for token-level EOS detection against
    /// <see cref="StreamingOptions.EosTokenIds"/> (it holds the sampled token); the base enforces
    /// the returned <c>IsEos</c> flag and <see cref="StreamingOptions.MaxSteps"/>.
    /// </summary>
    protected abstract (TStep Step, object? NextState, bool IsEos) ProcessStepOutputs(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? buildContext);

    /// <summary>Reduce the accumulated steps into the final one-shot <typeparamref name="TOutput"/>.</summary>
    protected abstract TOutput BuildFinalOutput(IReadOnlyList<TStep> steps);

    /// <summary>
    /// Samples one token id from a vocab-length logit row. Greedy argmax when
    /// <see cref="StreamingOptions.Sampling"/>'s temperature is disabled (0); otherwise
    /// temperature-scaled softmax sampling with optional top-k / top-p filtering.
    /// Engines call this from <see cref="ProcessStepOutputs"/> when they want the default sampler.
    /// </summary>
    protected virtual int SampleNextToken(ReadOnlySpan<float> logits)
    {
        if (logits.IsEmpty) return -1;

        var sampling = CurrentOptions?.Sampling;
        if (sampling is null || sampling.Temperature <= 0f)
            return Argmax(logits);

        // Temperature scaling.
        float temperature = sampling.Temperature;
        int vocab = logits.Length;

        // Rent vocab-sized buffers instead of allocating them on every step. For a typical
        // vocabulary (e.g. Phi-3's 32,006) a fresh double[32006] would land on the Large Object Heap
        // each token, causing severe GC pressure during generation. Rented arrays may be larger than
        // `vocab`, so every loop below is bounded by `vocab` and helpers take a vocab-length span.
        // Declare the rented arrays as nullable outside the try so a rent that throws cannot leak an
        // already-rented buffer; the finally block returns whichever buffers were successfully rented.
        float[]? scaled = null;
        double[]? probs = null;
        int[]? order = null;
        try
        {
            scaled = ArrayPool<float>.Shared.Rent(vocab);
            probs = ArrayPool<double>.Shared.Rent(vocab);

            for (int i = 0; i < vocab; i++)
                scaled[i] = logits[i] / temperature;

            // Top-k filtering: keep only the k highest logits, mask the rest to -inf.
            int topK = sampling.TopK;
            if (topK > 0 && topK < vocab)
            {
                float cutoff = NthLargest(scaled.AsSpan(0, vocab), topK);
                for (int i = 0; i < vocab; i++)
                    if (scaled[i] < cutoff)
                        scaled[i] = float.NegativeInfinity;
            }

            // Softmax (numerically stable: subtract max of finite values).
            float max = float.NegativeInfinity;
            for (int i = 0; i < vocab; i++)
                if (scaled[i] > max) max = scaled[i];
            if (float.IsNegativeInfinity(max))
                return Argmax(logits); // everything masked; fall back to greedy

            double sum = 0d;
            for (int i = 0; i < vocab; i++)
            {
                double p = Math.Exp(scaled[i] - max);
                probs[i] = p;
                sum += p;
            }

            // Top-p (nucleus) filtering: keep the smallest set of highest-prob tokens whose
            // cumulative probability reaches p, sorted descending.
            float topP = sampling.TopP;
            if (topP > 0f && topP < 1f)
            {
                order = ArrayPool<int>.Shared.Rent(vocab);
                for (int i = 0; i < vocab; i++) order[i] = i;
                // Sort token indices by descending probability without per-step allocation: a struct
                // comparer is specialized by MemoryExtensions.Sort and never boxes, unlike
                // Comparer<int>.Create(lambda), which allocates a Comparer + delegate every token.
                order.AsSpan(0, vocab).Sort(new DescendingProbabilityIndexComparer(probs));
                double cum = 0d;
                bool reached = false;
                for (int i = 0; i < vocab; i++)
                {
                    if (reached)
                    {
                        probs[order[i]] = 0d;
                    }
                    else
                    {
                        cum += probs[order[i]] / sum;
                        if (cum >= topP) reached = true;
                    }
                }
                // Renormalize the kept set.
                double keptSum = 0d;
                for (int i = 0; i < vocab; i++) keptSum += probs[i];
                if (keptSum <= 0d) return Argmax(logits);
                for (int i = 0; i < vocab; i++) probs[i] /= keptSum;
            }
            else
            {
                for (int i = 0; i < vocab; i++) probs[i] /= sum;
            }

            // Sample from the resulting distribution.
            double r = Random.Shared.NextDouble();
            double acc = 0d;
            for (int i = 0; i < vocab; i++)
            {
                acc += probs[i];
                if (r < acc) return i;
            }
            return vocab - 1;
        }
        finally
        {
            if (scaled is not null)
                ArrayPool<float>.Shared.Return(scaled);
            if (probs is not null)
                ArrayPool<double>.Shared.Return(probs);
            if (order is not null)
                ArrayPool<int>.Shared.Return(order);
        }
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
                best = i;
            }
        }
        return best;
    }

    /// <summary>Returns the cutoff value of the k-th largest element (allocation-free partial sort).
    /// Maintains a sorted span of the k largest values seen; its minimum is the k-th largest.
    /// Operates on a vocab-length slice of a (possibly larger) rented array. Allocation-free (uses
    /// stackalloc for small k, an ArrayPool rental otherwise) so the hot sampling path never hits the
    /// managed heap — the previous PriorityQueue approach allocated on every token-generation step.</summary>
    private static float NthLargest(ReadOnlySpan<float> values, int k)
    {
        float[]? rented = null;
        Span<float> temp = k <= 256 ? stackalloc float[k] : (rented = ArrayPool<float>.Shared.Rent(k));
        try
        {
            temp.Fill(float.NegativeInfinity);
            for (int i = 0; i < values.Length; i++)
            {
                float v = values[i];
                if (v <= temp[0]) continue;

                // Insert v into the ascending-sorted top-k window, evicting the current minimum
                // (temp[0]). BinarySearch gives the insertion point; shift the sub-slice left by one
                // to drop the old minimum and place v at the now-vacated slot.
                int index = temp.BinarySearch(v);
                if (index < 0) index = ~index;
                if (index > 1)
                    temp.Slice(1, index - 1).CopyTo(temp.Slice(0, index - 1));
                if (index > 0)
                    temp[index - 1] = v;
            }
            return temp[0];
        }
        finally
        {
            if (rented is not null)
                ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Compares two array indices by the value they reference in descending order. A struct so that
    /// the span sort helper can specialize it without boxing or any heap allocation (a delegate-based
    /// comparer would allocate on every token-generation step).
    /// </summary>
    private readonly struct DescendingProbabilityIndexComparer : IComparer<int>
    {
        private readonly double[] _values;

        public DescendingProbabilityIndexComparer(double[] values) => _values = values;

        public int Compare(int a, int b) => _values[b].CompareTo(_values[a]);
    }

    /// <summary>
    /// Run one forward pass. Virtual so tests can substitute a fake session without a real model.
    /// </summary>
    protected virtual IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunStep(
        IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        var session = _session ?? throw new InvalidOperationException("Engine is not loaded.");
        return session.Run(inputs, session.OutputNames, _runOptions);
    }

    /// <summary>
    /// Clone a <see cref="DenseTensor{T}"/> of <see cref="float"/> carried in a model output so it
    /// survives disposal of the output collection, renaming it from a <c>present_*</c> name to the
    /// matching <c>past_*</c> name for the next step. Engines call this from
    /// <see cref="ProcessStepOutputs"/> to carry the KV cache forward. Other dtypes require extension.
    /// </summary>
    protected static NamedOnnxValue? CloneAsPastKeyValue(DisposableNamedOnnxValue output, string pastName)
    {
        if (output.Value is not DenseTensor<float> src) return null;
        var copy = new DenseTensor<float>(src.Dimensions);
        src.Buffer.Span.CopyTo(copy.Buffer.Span);
        return NamedOnnxValue.CreateFromTensor(pastName, copy);
    }

    /// <summary>
    /// Convenience <see cref="IDisposable"/> holder for a list of carry-forward KV-cache tensors.
    /// Engines that use <see cref="CloneAsPastKeyValue"/> should wrap the cloned values in an
    /// instance of this type (or any other <see cref="IDisposable"/>) and return it from
    /// <see cref="ProcessStepOutputs"/> as the next state. The base loop disposes the previous
    /// state on step rollover and the final state on stream end, releasing the cloned tensors.
    /// </summary>
    protected sealed class KeyValueCacheState : IDisposable
    {
        private readonly List<NamedOnnxValue> _values;
        private bool _disposed;

        public KeyValueCacheState(List<NamedOnnxValue> values)
        {
            _values = values ?? throw new ArgumentNullException(nameof(values));
        }

        /// <summary>The carry-forward KV-cache tensors.</summary>
        public IReadOnlyList<NamedOnnxValue> Values => _values;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            DisposeNamedOnnxValues(_values);
        }
    }

    /// <summary>Dispose any <see cref="NamedOnnxValue"/> items in <paramref name="values"/> that are disposable.</summary>
    protected static void DisposeNamedOnnxValues(IReadOnlyCollection<NamedOnnxValue> values)
    {
        foreach (var t in values)
        {
            if (t is IDisposable disposable)
                disposable.Dispose();
        }
    }

    // ---- One-shot pre/post hooks are unused on streaming engines (PredictAsync is
    //      overridden to route through the stream). Provide stubs so concrete streaming
    //      engines are not forced to implement dead methods.
    /// <summary>Not used by streaming engines; one-shot <see cref="PredictAsync"/> is routed through the stream.</summary>
    protected sealed override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(TInput input) =>
        throw new NotSupportedException(
            "Streaming engines route one-shot PredictAsync through PredictStreamingAsync; " +
            "PreProcessWithContext is not used. Override BuildStepInputs instead.");

    /// <summary>Not used by streaming engines.</summary>
    protected sealed override TOutput PostProcessWithContext(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context) =>
        throw new NotSupportedException(
            "Streaming engines route one-shot PredictAsync through PredictStreamingAsync; " +
            "PostProcessWithContext is not used. Override ProcessStepOutputs instead.");
}
