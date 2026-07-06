using InferenceEngine.Core;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Xunit;

namespace InferenceEngine.Tests;

public sealed class StreamingInferenceEngineTests : IDisposable
{
    private readonly List<BaseStreamingInferenceEngine<int, IReadOnlyList<int>, int>> _engines = [];

    [Fact]
    public async Task Stream_YieldsExpectedTokens()
    {
        var tokens = new[] { 11, 22, 33, 44, 55 };
        var engine = CreateEngine(tokens);

        var streamed = await ToListAsync(engine.PredictStreamingAsync(tokens.Length));

        Assert.Equal(tokens, streamed);
    }

    [Fact]
    public async Task PredictAsync_Matches_PredictStreamingAsync_Parity()
    {
        var tokens = new[] { 7, 8, 9, 10, 11, 12 };
        var engine = CreateEngine(tokens);

        var oneShot = await engine.PredictAsync(tokens.Length);
        var streamed = await ToListAsync(engine.PredictStreamingAsync(tokens.Length));

        Assert.Equal(oneShot, streamed);
    }

    [Fact]
    public async Task MaxSteps_StopsLoop()
    {
        var tokens = new[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var engine = CreateEngine(tokens);

        var options = new StreamingOptions { MaxSteps = 3 };

        var streamed = await ToListAsync(engine.PredictStreamingAsync(tokens.Length, options));

        Assert.Equal(3, streamed.Count);
        Assert.Equal(new[] { 1, 2, 3 }, streamed);
    }

    [Fact]
    public async Task Eos_StopsLoop()
    {
        const int eos = 99;
        var tokens = new[] { 10, 20, eos, 30, 40 };
        var engine = CreateEngine(tokens);

        var options = new StreamingOptions { EosTokenIds = [eos] };

        var streamed = await ToListAsync(engine.PredictStreamingAsync(tokens.Length, options));

        // The EOS token is yielded as the terminating step; generation does not continue past it.
        Assert.Equal(new[] { 10, 20, eos }, streamed);
    }

    [Fact]
    public async Task Cancellation_AbortsAndLeavesClean()
    {
        var tokens = new[] { 1, 2, 3, 4, 5 };
        var engine = CreateEngine(tokens);

        using var cts = new CancellationTokenSource();
        IAsyncEnumerator<int> e = engine
            .PredictStreamingAsync(tokens.Length, null, cts.Token)
            .GetAsyncEnumerator(cts.Token);
        try
        {
            Assert.True(await e.MoveNextAsync()); // step 0
            Assert.Equal(1, e.Current);

            cts.Cancel();

            // Cancellation is honored at the next step boundary (step 1).
            await Assert.ThrowsAnyAsync<OperationCanceledException>(
                async () => await e.MoveNextAsync());
        }
        finally
        {
            // Disposing the enumerator runs the iterator's finally, releasing the in-flight guard.
            await e.DisposeAsync();
        }

        // A subsequent stream on the same instance must work (guard was released).
        var subsequent = await ToListAsync(engine.PredictStreamingAsync(tokens.Length));
        Assert.Equal(tokens, subsequent);
    }

    [Fact]
    public async Task Concurrency_SecondStreamRejected()
    {
        var tokens = new[] { 1, 2, 3, 4, 5 };
        var engine = CreateEngine(tokens);

        IAsyncEnumerator<int> first = engine.PredictStreamingAsync(tokens.Length).GetAsyncEnumerator();
        try
        {
            Assert.True(await first.MoveNextAsync()); // acquires the single-in-flight guard

            // A second concurrent stream on the same instance is rejected.
            await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            {
                await foreach (var _ in engine.PredictStreamingAsync(tokens.Length))
                {
                    // unreachable: the guard check throws on the first MoveNextAsync
                }
            });
        }
        finally
        {
            await first.DisposeAsync(); // release the guard
        }
    }

    [Fact]
    public async Task Disposal_EveryStepOutputDisposed()
    {
        var tokens = new[] { 1, 2, 3, 4, 5 };
        var engine = CreateEngine(tokens);

        var streamed = await ToListAsync(engine.PredictStreamingAsync(tokens.Length));

        Assert.Equal(tokens, streamed);
        Assert.Equal(tokens.Length, engine.OutputCollectionsDisposed);
    }

    // T3 (H4): the carry-forward state is IDisposable and the base loop disposes the prior state on
    // step rollover and the final state on EOS — no KV-cache leak across a generation.
    [Fact]
    public async Task KvCarryState_DisposedOnRolloverAndEos()
    {
        var tokens = new[] { 10, 20, 30 };
        var engine = new CarryTrackingEngine(tokens, new InferenceEngineOptions());
        _engines.Add(engine);

        var streamed = await ToListAsync(engine.PredictStreamingAsync(tokens.Length));

        Assert.Equal(tokens, streamed);
        // One state per step; all are disposed (prior on rollover + final on EOS).
        Assert.Equal(tokens.Length, engine.StatesCreated);
        Assert.Equal(tokens.Length, engine.StatesDisposed);
    }

    // T3 (H4): the carry state is disposed when the stream throws mid-generation.
    [Fact]
    public async Task KvCarryState_DisposedOnException()
    {
        var engine = new CarryTrackingEngine(new[] { 1, 2, 3 }, new InferenceEngineOptions(), throwOnStep: 1);
        _engines.Add(engine);

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await ToListAsync(engine.PredictStreamingAsync(3));
        });

        // The step-0 carry state must have been disposed in the iterator's finally.
        Assert.Equal(1, engine.StatesCreated);
        Assert.Equal(1, engine.StatesDisposed);
    }

    // T3 (H4): the carry state is disposed on cancellation too.
    [Fact]
    public async Task KvCarryState_DisposedOnCancellation()
    {
        var engine = new CarryTrackingEngine(new[] { 1, 2, 3, 4 }, new InferenceEngineOptions());
        _engines.Add(engine);

        using var cts = new CancellationTokenSource();
        IAsyncEnumerator<int> e = engine
            .PredictStreamingAsync(4, null, cts.Token).GetAsyncEnumerator(cts.Token);
        try
        {
            Assert.True(await e.MoveNextAsync()); // step 0 (creates a carry state)
            cts.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await e.MoveNextAsync());
        }
        finally
        {
            await e.DisposeAsync();
        }

        Assert.Equal(1, engine.StatesCreated);
        Assert.Equal(1, engine.StatesDisposed);
    }

    // T5: a partially-consumed enumerator that is properly disposed releases the single-in-flight
    // guard, so a subsequent stream on the same instance works.
    [Fact]
    public async Task PartiallyConsumedEnumerator_DisposeReleasesInFlightGuard()
    {
        var tokens = new[] { 1, 2, 3, 4, 5 };
        var engine = CreateEngine(tokens);

        IAsyncEnumerator<int> e = engine.PredictStreamingAsync(tokens.Length).GetAsyncEnumerator();
        try
        {
            Assert.True(await e.MoveNextAsync());
            Assert.Equal(1, e.Current);
            // Abandon early (but dispose properly per the streaming contract).
        }
        finally
        {
            await e.DisposeAsync();
        }

        // The guard was released; a fresh stream runs to completion.
        var full = await ToListAsync(engine.PredictStreamingAsync(tokens.Length));
        Assert.Equal(tokens, full);
    }

    // M3: temperature > 0 yields non-argmax sampling (the SamplingParams config is now honored).
    [Fact]
    public async Task Sampling_TemperaturePositive_ProducesNonArgmax()
    {
        var engine = new SamplingEngine(new InferenceEngineOptions());
        _engines.Add(engine);

        // Uniform logits [5, 5, 5]: argmax is 0, but temperature sampling must sometimes pick 1 or 2.
        var options = new StreamingOptions { MaxSteps = 60, Sampling = new SamplingParams { Temperature = 1f } };

        var sampled = await ToListAsync(engine.PredictStreamingAsync(0, options));

        Assert.NotEmpty(sampled);
        Assert.Contains(sampled, t => t != 0);
    }

    // --- Helpers ---

    private FakeStreamingEngine CreateEngine(IReadOnlyList<int> tokens)
    {
        var engine = new FakeStreamingEngine(tokens, new InferenceEngineOptions());
        _engines.Add(engine);
        return engine;
    }

    private static async Task<List<T>> ToListAsync<T>(IAsyncEnumerable<T> source)
    {
        var list = new List<T>();
        await foreach (var item in source.ConfigureAwait(false))
            list.Add(item);
        return list;
    }

    public void Dispose()
    {
        foreach (var engine in _engines)
            engine.Dispose();
    }

    // --- Fake streaming engine (no real ONNX session required) ---
    //
    // TInput  = int           (prompt length, unused by the fake)
    // TOutput = IReadOnlyList<int> (the fully reduced token list)
    // TStep   = int           (one sampled token id per yield)

    private sealed class FakeStreamingEngine : BaseStreamingInferenceEngine<int, IReadOnlyList<int>, int>
    {
        private readonly IReadOnlyList<int> _tokens;

        public FakeStreamingEngine(IReadOnlyList<int> tokens, InferenceEngineOptions options)
            : base(options, NullLogger<FakeStreamingEngine>.Instance)
        {
            _tokens = tokens;
        }

        // Counts how many per-step output collections the base loop disposed.
        public int OutputCollectionsDisposed { get; private set; }

        protected override string ModelName => "fake-streaming.onnx";

        // Bypass real session construction entirely; the RunStep seam does not need it.
        public override Task LoadAsync() => Task.CompletedTask;

        // Never call the base; return a tiny disposable fake output collection.
        protected override IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunStep(
            IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            return new FakeOutputCollection(() => OutputCollectionsDisposed++);
        }

        // Carry the step index through BuildContext so ProcessStepOutputs can index _tokens.
        protected override (List<NamedOnnxValue> Inputs, object? BuildContext) BuildStepInputs(
            int input, object? state, int step)
            => (new List<NamedOnnxValue>(), step);

        protected override (int Step, object? NextState, bool IsEos) ProcessStepOutputs(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? buildContext)
        {
            int step = (int)buildContext!;
            int token = _tokens[step];
            // Terminate naturally at the last token, or when an EOS id is sampled.
            bool isEos = step >= _tokens.Count - 1 || IsEosToken(token);
            return (token, null, isEos);
        }

        protected override IReadOnlyList<int> BuildFinalOutput(IReadOnlyList<int> steps)
            => steps;

        private bool IsEosToken(int token)
            => CurrentOptions?.EosTokenIds is { Count: > 0 } eos && eos.Contains(token);
    }

    // Minimal empty IDisposableReadOnlyCollection<DisposableNamedOnnxValue>. ProcessStepOutputs
    // never inspects real tensors, so an empty backing array is sufficient.
    private sealed class FakeOutputCollection : IDisposableReadOnlyCollection<DisposableNamedOnnxValue>
    {
        private static readonly DisposableNamedOnnxValue[] _empty = [];
        private readonly Action _onDispose;

        public FakeOutputCollection(Action onDispose)
        {
            _onDispose = onDispose;
        }

        public int Count => _empty.Length;

        public DisposableNamedOnnxValue this[int index] => _empty[index];

        public IEnumerator<DisposableNamedOnnxValue> GetEnumerator()
            => ((IEnumerable<DisposableNamedOnnxValue>)_empty).GetEnumerator();

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            => _empty.GetEnumerator();

        public void Dispose() => _onDispose();
    }

    // A streaming engine that returns a disposable carry state each step, so the base loop's
    // KV-cache disposal contract (H4) can be observed. (No real ONNX tensors needed: the contract
    // under test is "prior state disposed on rollover, final state disposed on stream end".)
    private sealed class CarryTrackingEngine : BaseStreamingInferenceEngine<int, IReadOnlyList<int>, int>
    {
        private readonly IReadOnlyList<int> _tokens;
        private readonly int _throwOnStep;

        public int StatesCreated { get; private set; }
        public int StatesDisposed { get; private set; }

        public CarryTrackingEngine(IReadOnlyList<int> tokens, InferenceEngineOptions options, int throwOnStep = -1)
            : base(options, NullLogger<CarryTrackingEngine>.Instance)
        {
            _tokens = tokens;
            _throwOnStep = throwOnStep;
        }

        protected override string ModelName => "carry-track.onnx";
        public override Task LoadAsync() => Task.CompletedTask;

        protected override IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunStep(
            IReadOnlyCollection<NamedOnnxValue> inputs)
            => new FakeOutputCollection(() => { });

        protected override (List<NamedOnnxValue> Inputs, object? BuildContext) BuildStepInputs(
            int input, object? state, int step)
            => (new List<NamedOnnxValue>(), step);

        protected override (int Step, object? NextState, bool IsEos) ProcessStepOutputs(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? buildContext)
        {
            int step = (int)buildContext!;
            if (step == _throwOnStep)
                throw new InvalidOperationException("simulated mid-stream failure");

            int token = _tokens[step];
            bool isEos = step >= _tokens.Count - 1;
            StatesCreated++;
            var state = new TrackingCarryState(() => StatesDisposed++);
            return (token, state, isEos);
        }

        protected override IReadOnlyList<int> BuildFinalOutput(IReadOnlyList<int> steps) => steps;
    }

    // Disposable carry state that reports its disposal, mirroring the KeyValueCacheState contract.
    private sealed class TrackingCarryState : IDisposable
    {
        private readonly Action _onDispose;
        private bool _disposed;

        public TrackingCarryState(Action onDispose) => _onDispose = onDispose;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _onDispose();
        }
    }

    // A streaming engine that emits uniform logits and routes sampling through the base
    // SampleNextToken, so M3's temperature/top-k/top-p honoring is observable.
    private sealed class SamplingEngine : BaseStreamingInferenceEngine<int, IReadOnlyList<int>, int>
    {
        public SamplingEngine(InferenceEngineOptions options)
            : base(options, NullLogger<SamplingEngine>.Instance) { }

        protected override string ModelName => "sampling.onnx";
        public override Task LoadAsync() => Task.CompletedTask;

        protected override IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunStep(
            IReadOnlyCollection<NamedOnnxValue> inputs)
            => new FakeOutputCollection(() => { });

        protected override (List<NamedOnnxValue> Inputs, object? BuildContext) BuildStepInputs(
            int input, object? state, int step)
            => (new List<NamedOnnxValue>(), step);

        protected override (int Step, object? NextState, bool IsEos) ProcessStepOutputs(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? buildContext)
        {
            int step = (int)buildContext!;
            // Uniform logits over a 3-token vocab; argmax is 0.
            Span<float> logits = stackalloc float[3] { 5f, 5f, 5f };
            int sampled = SampleNextToken(logits);
            int maxSteps = CurrentOptions?.MaxSteps ?? 256;
            bool isEos = step >= maxSteps - 1;
            return (sampled, null, isEos);
        }

        protected override IReadOnlyList<int> BuildFinalOutput(IReadOnlyList<int> steps) => steps;
    }
}
