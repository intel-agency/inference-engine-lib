using InferenceEngine.Core;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace InferenceEngine.Tests;

/// <summary>
/// Streaming-engine coverage that needs a real ONNX session or the base sampling/KV-cache
/// plumbing: <see cref="BaseStreamingInferenceEngine{TInput,TOutput,TStep}.RunStep"/> against a
/// hand-built identity model, greedy (temperature-disabled) sampling, KV-cache state clone /
/// restart / double-dispose, non-float output cloning, input-value disposal, and the sealed
/// one-shot pre/post stubs.
/// </summary>
public sealed class StreamingInferenceEngineRealSessionTests : IDisposable
{
    private readonly string _testDir;
    private readonly List<IDisposable> _disposables = [];

    public StreamingInferenceEngineRealSessionTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"stream-real-{Guid.NewGuid()}");
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
        try { Directory.Delete(_testDir, true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }

    [SkippableFact]
    public async Task RealSessionStream_RunsStepsAndCarriesKvCacheState()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateRealSessionEngine(floatLogit: 4f, steps: 2);

        var streamed = await ToListAsync(engine.PredictStreamingAsync("prompt"));

        // Greedy sampling over a single-logit row: every step samples token 0.
        Assert.Equal([0, 0], streamed);
        // One KV-cache state per step, cloned from the step output and disposed by the base loop.
        Assert.Equal(2, engine.CacheStatesCreated);
        Assert.Equal(2, engine.CacheStatesDisposed);
        Assert.True(engine.CloneSucceeded);
    }

    [SkippableFact]
    public async Task RealSessionStream_CancellationMidToken_ReleasesStateAndGuard()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateRealSessionEngine(floatLogit: 4f, steps: 5);
        using var cts = new CancellationTokenSource();

        IAsyncEnumerator<int> e = engine
            .PredictStreamingAsync("prompt", null, cts.Token)
            .GetAsyncEnumerator(cts.Token);
        try
        {
            Assert.True(await e.MoveNextAsync()); // step 0 (creates + carries KV state)
            cts.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await e.MoveNextAsync());
        }
        finally
        {
            await e.DisposeAsync();
        }

        // The mid-stream cancellation disposed the carried KV state and released the guard.
        Assert.Equal(1, engine.CacheStatesCreated);
        Assert.Equal(1, engine.CacheStatesDisposed);

        var subsequent = await ToListAsync(engine.PredictStreamingAsync("prompt"));
        Assert.NotEmpty(subsequent); // stream restarts cleanly with fresh KV cache
    }

    [SkippableFact]
    public async Task RealSessionStream_NonFloatOutput_CloneAsPastKeyValueReturnsNull()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = new IntOutputStreamingEngine(
            new InferenceEngineOptions
            {
                UseGpuAcceleration = false,
                WarmupOnLoad = false,
                ModelProvider = new Core.ModelProviders.LocalModelProvider(WriteModel(MinimalOnnxModel.ElementType.Int32)),
            });
        _disposables.Add(engine);

        var streamed = await ToListAsync(engine.PredictStreamingAsync("prompt"));

        Assert.Single(streamed); // terminates after the un-cloneable step
        Assert.False(engine.CloneSucceeded); // non-float tensor cannot be carried forward
    }

    [Fact]
    public async Task SampleNextToken_GreedyWhenSamplingDisabled_ReturnsArgmax()
    {
        var engine = new FixedLogitsStreamingEngine([1f, 3f, 9f, 2f]);
        _disposables.Add(engine);

        // Default StreamingOptions carries no Sampling config -> greedy argmax over the row.
        var streamed = await ToListAsync(engine.PredictStreamingAsync("prompt"));
        Assert.Equal([2], streamed);
    }

    [Fact]
    public async Task SampleNextToken_EmptyLogits_YieldsSentinelAndTerminates()
    {
        var engine = new FixedLogitsStreamingEngine([]);
        _disposables.Add(engine);

        var streamed = await ToListAsync(engine.PredictStreamingAsync("prompt"));
        Assert.Equal([-1], streamed); // sampler rejects the empty row deterministically
    }

    [Fact]
    public async Task SampleNextToken_AllNanLogits_FallsBackToArgmax()
    {
        var engine = new FixedLogitsStreamingEngine([float.NaN, float.NaN, float.NaN]);
        _disposables.Add(engine);
        var options = new StreamingOptions
        {
            Sampling = new SamplingParams { Temperature = 1f, TopK = 2 },
        };

        // Top-k masks nothing (NaN comparisons are false), so max stays -inf and the sampler
        // must fall back to greedy instead of sampling from a degenerate distribution.
        var streamed = await ToListAsync(engine.PredictStreamingAsync("prompt", options));
        Assert.Equal([0], streamed);
    }

    [Fact]
    public void KeyValueCacheState_ExposesValuesAndDoubleDisposeIsNoOp()
    {
        var engine = CreateFakeEngine();
        var tensor = NamedOnnxValue.CreateFromTensor("past_0", new DenseTensor<float>([1]));

        Assert.Equal(1, engine.KvStateValuesCount([tensor]));
        engine.KvStateDoubleDisposeIsNoOp([tensor]); // must not throw or re-dispose the values
    }

    [Fact]
    public void KeyValueCacheState_NullValues_ThrowsArgumentNullException()
    {
        var engine = CreateFakeEngine();
        engine.KvStateNullValuesThrows();
    }

    [Fact]
    public void DisposeNamedOnnxValues_SkipsNonDisposableValues()
    {
        var engine = CreateFakeEngine();

        // NamedOnnxValue (non-disposable) must be skipped without throwing; an empty list is fine.
        engine.ExposedDisposeNamedOnnxValues([]);
        engine.ExposedDisposeNamedOnnxValues([NamedOnnxValue.CreateFromTensor("a", new DenseTensor<float>([1]))]);
    }

    [Fact]
    public void OneShotPreAndPostStubs_ThrowNotSupportedException()
    {
        var engine = CreateFakeEngine();

        Assert.Throws<NotSupportedException>(() => engine.ExposedPreProcessWithContext("x"));
        Assert.Throws<NotSupportedException>(() => engine.ExposedPostProcessWithContext(null!, null));
    }

    // --- Helpers ---

    private static async Task<List<T>> ToListAsync<T>(IAsyncEnumerable<T> source)
    {
        var list = new List<T>();
        await foreach (var item in source.ConfigureAwait(false))
            list.Add(item);
        return list;
    }

    private string WriteModel(MinimalOnnxModel.ElementType elementType = MinimalOnnxModel.ElementType.Float)
    {
        var path = Path.Combine(_testDir, $"model-{Guid.NewGuid():N}.onnx");
        MinimalOnnxModel.Build(path, elementType);
        return path;
    }

    private RealSessionStreamingEngine CreateRealSessionEngine(float floatLogit, int steps)
    {
        var engine = new RealSessionStreamingEngine(
            floatLogit,
            steps,
            new InferenceEngineOptions
            {
                UseGpuAcceleration = false,
                WarmupOnLoad = false,
                ModelProvider = new Core.ModelProviders.LocalModelProvider(WriteModel()),
            });
        _disposables.Add(engine);
        return engine;
    }

    private FakeStreamingEngine CreateFakeEngine()
    {
        var engine = new FakeStreamingEngine();
        _disposables.Add(engine);
        return engine;
    }

    private static bool IsNativeOnnxRuntimeAvailable()
    {
        try
        {
            using var opts = new SessionOptions();
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Streaming engine whose BuildStepInputs feeds a real identity model and whose
    /// ProcessStepOutputs routes through the base sampler and KV-cache clone helpers, with a
    /// <see cref="BaseStreamingInferenceEngine{TInput,TOutput,TStep}.KeyValueCacheState"/> carry state.
    /// </summary>
    private sealed class RealSessionStreamingEngine : BaseStreamingInferenceEngine<string, IReadOnlyList<int>, int>
    {
        private readonly float _logit;
        private readonly int _maxSteps;
        private int _step;

        public int CacheStatesCreated { get; private set; }
        public int CacheStatesDisposed { get; private set; }
        public bool CloneSucceeded { get; private set; }

        public RealSessionStreamingEngine(float logit, int maxSteps, InferenceEngineOptions options)
            : base(options, NullLogger<RealSessionStreamingEngine>.Instance)
        {
            _logit = logit;
            _maxSteps = maxSteps;
        }

        protected override string ModelName => "real-stream.onnx";

        protected override (List<NamedOnnxValue> Inputs, object? BuildContext) BuildStepInputs(
            string input, object? state, int step)
        {
            // Prefill: fresh input row. Decode: the previous step's cloned cache rides in `state`.
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new[] { 1 }) { [0] = _logit }),
            };
            return (inputs, step);
        }

        protected override (int Step, object? NextState, bool IsEos) ProcessStepOutputs(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? buildContext)
        {
            var first = outputs.First();
            var cloned = CloneAsPastKeyValue(first, "past_0");
            CloneSucceeded = cloned is not null;

            float[] logits = [_logit];
            int token = SampleNextToken(logits); // greedy: no Sampling config on default options

            CacheStatesCreated++;
            var nextState = new CountingCacheState(() => CacheStatesDisposed++, cloned is null ? [] : [cloned]);
            bool isEos = ++_step >= _maxSteps;
            return (token, nextState, isEos);
        }

        protected override IReadOnlyList<int> BuildFinalOutput(IReadOnlyList<int> steps) => steps;

        private sealed class CountingCacheState : IDisposable
        {
            private readonly Action _onDispose;
            private readonly List<NamedOnnxValue> _values;
            private bool _disposed;

            public CountingCacheState(Action onDispose, List<NamedOnnxValue> values)
            {
                _onDispose = onDispose;
                _values = values;
            }

            public void Dispose()
            {
                if (_disposed) return;
                _disposed = true;
                BaseStreamingInferenceEngine<string, IReadOnlyList<int>, int>.DisposeNamedOnnxValues(_values);
                _onDispose();
            }
        }
    }

    /// <summary>Engine over an int-output identity model: CloneAsPastKeyValue must return null.</summary>
    private sealed class IntOutputStreamingEngine : BaseStreamingInferenceEngine<string, IReadOnlyList<int>, int>
    {
        public bool CloneSucceeded { get; private set; }

        public IntOutputStreamingEngine(InferenceEngineOptions options)
            : base(options, NullLogger<IntOutputStreamingEngine>.Instance) { }

        protected override string ModelName => "int-stream.onnx";

        protected override (List<NamedOnnxValue> Inputs, object? BuildContext) BuildStepInputs(
            string input, object? state, int step)
            => ([NamedOnnxValue.CreateFromTensor("input", new DenseTensor<int>(new[] { 1 }) { [0] = 0 })], step);

        protected override (int Step, object? NextState, bool IsEos) ProcessStepOutputs(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? buildContext)
        {
            CloneSucceeded = CloneAsPastKeyValue(outputs.First(), "past_0") is not null;
            return (0, null, IsEos: true); // terminate after the un-cloneable step
        }

        protected override IReadOnlyList<int> BuildFinalOutput(IReadOnlyList<int> steps) => steps;
    }

    /// <summary>
    /// Streaming engine that routes fixed logits through the base <c>SampleNextToken</c> so the
    /// sampler's greedy / degenerate-distribution branches are exercised under real
    /// <see cref="BaseStreamingInferenceEngine{TInput,TOutput,TStep}.CurrentOptions"/>.
    /// </summary>
    private sealed class FixedLogitsStreamingEngine : BaseStreamingInferenceEngine<string, IReadOnlyList<int>, int>
    {
        private readonly float[] _logits;

        public FixedLogitsStreamingEngine(float[] logits)
            : base(new InferenceEngineOptions(), NullLogger<FixedLogitsStreamingEngine>.Instance)
        {
            _logits = logits;
        }

        protected override string ModelName => "fixed-logits.onnx";
        public override Task LoadAsync() => Task.CompletedTask;

        protected override IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunStep(
            IReadOnlyCollection<NamedOnnxValue> inputs)
            => new EmptyOutputCollection();

        protected override (List<NamedOnnxValue> Inputs, object? BuildContext) BuildStepInputs(
            string input, object? state, int step) => ([], step);

        protected override (int Step, object? NextState, bool IsEos) ProcessStepOutputs(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? buildContext)
            => (SampleNextToken(_logits), null, IsEos: true); // single step, then EOS

        protected override IReadOnlyList<int> BuildFinalOutput(IReadOnlyList<int> steps) => steps;
    }

    private sealed class EmptyOutputCollection : IDisposableReadOnlyCollection<DisposableNamedOnnxValue>
    {
        private static readonly DisposableNamedOnnxValue[] Empty = [];

        public int Count => Empty.Length;
        public DisposableNamedOnnxValue this[int index] => Empty[index];
        public IEnumerator<DisposableNamedOnnxValue> GetEnumerator() => ((IEnumerable<DisposableNamedOnnxValue>)Empty).GetEnumerator();
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => Empty.GetEnumerator();
        public void Dispose() { }
    }

    /// <summary>Minimal engine exposing the protected utility surface for direct tests.</summary>
    private sealed class FakeStreamingEngine : BaseStreamingInferenceEngine<string, IReadOnlyList<int>, int>
    {
        public FakeStreamingEngine() : base(new InferenceEngineOptions(), NullLogger<FakeStreamingEngine>.Instance) { }

        protected override string ModelName => "fake-expose.onnx";
        public override Task LoadAsync() => Task.CompletedTask;

        protected override (List<NamedOnnxValue> Inputs, object? BuildContext) BuildStepInputs(
            string input, object? state, int step) => ([], step);

        protected override (int Step, object? NextState, bool IsEos) ProcessStepOutputs(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? buildContext)
            => (0, null, true);

        protected override IReadOnlyList<int> BuildFinalOutput(IReadOnlyList<int> steps) => steps;

        // KeyValueCacheState is a protected nested type: expose its contract as plain observables.
        public int KvStateValuesCount(List<NamedOnnxValue> values)
        {
            using var state = new KeyValueCacheState(values);
            return state.Values.Count;
        }

        public void KvStateDoubleDisposeIsNoOp(List<NamedOnnxValue> values)
        {
            var state = new KeyValueCacheState(values);
            state.Dispose();
            state.Dispose();
        }

        public void KvStateNullValuesThrows() => Assert.Throws<ArgumentNullException>(() => new KeyValueCacheState(null!));

        public void ExposedDisposeNamedOnnxValues(IReadOnlyCollection<NamedOnnxValue> values)
            => DisposeNamedOnnxValues(values);

        public (List<NamedOnnxValue>, object?) ExposedPreProcessWithContext(string input) => PreProcessWithContext(input);

        public IReadOnlyList<int> ExposedPostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
            => PostProcessWithContext(output, context);
    }
}
