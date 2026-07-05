using System;
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

                state = nextState;
                yield return yielded;

                if (isEos) yield break;
            }
        }
        finally
        {
            // Reset per-run state and release the in-flight guard on every exit path
            // (success, EOS, MaxSteps, exception, and cancellation).
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
    /// Greedy (argmax) sampling over a logit row. Override for top-k/top-p/temperature/
    /// repetition-penalty using <see cref="StreamingOptions.Sampling"/>. The engine calls this from
    /// <see cref="ProcessStepOutputs"/> when it wants the default sampler.
    /// </summary>
    protected virtual int SampleNextToken(ReadOnlySpan<float> logits)
    {
        if (logits.IsEmpty) return -1;
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
