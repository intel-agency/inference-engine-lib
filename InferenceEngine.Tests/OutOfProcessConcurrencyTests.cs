using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using InferenceEngine.Core;
using InferenceEngine.Core.Diagnostics;
using InferenceEngine.Core.Worker;
using Microsoft.ML.OnnxRuntime;
using Xunit;

namespace InferenceEngine.Tests;

/// <summary>
/// Hermetic concurrency/timeout/disposal coverage for <see cref="OutOfProcessBaseInferenceEngine{TInput,TOutput}"/>
/// (review items H1, H2, M6). No real worker process is spawned: an in-memory duplex
/// <see cref="ChannelStream"/> pair stands in for the stdin/stdout pipe, and a real
/// <see cref="InferenceWorkerHost"/> loop (or a hang loop) plays the worker.
/// </summary>
public sealed class OutOfProcessConcurrencyTests
{
    // H1: two concurrent predicts each get their OWN correct result (no cross-pairing), and the
    // worker is spawned exactly once (no double-spawn on the auto-load path).
    [Fact]
    public async Task ConcurrentPredicts_DoNotCrossPair_AndSpawnOnce()
    {
        await using var engine = new TestOopEngine();
        await engine.LoadAsync();
        Assert.Equal(1, engine.SpawnCount);

        var a = engine.PredictAsync("A");
        var b = engine.PredictAsync("B");
        string[] results = await Task.WhenAll(a, b);

        Assert.Equal(new[] { "echo:A", "echo:B" }, results.OrderBy(x => x).ToArray());
        Assert.Equal(1, engine.SpawnCount);
    }

    // H1: concurrent loads do not double-spawn the worker.
    [Fact]
    public async Task ConcurrentLoads_SpawnWorkerOnce()
    {
        await using var engine = new TestOopEngine();

        await Task.WhenAll(engine.LoadAsync(), engine.LoadAsync(), engine.LoadAsync());

        Assert.Equal(1, engine.SpawnCount);
        Assert.True(engine.IsWorkerLoaded);
    }

    // H2: a worker that accepts load but never responds to predict causes a TimeoutException within
    // the configured inference timeout, instead of hanging indefinitely.
    [Fact]
    public async Task HungWorker_Predict_ThrowsTimeoutWithinDeadline()
    {
        await using var engine = new TestOopEngine(hang: true, inferenceTimeout: TimeSpan.FromMilliseconds(400));
        await engine.LoadAsync(); // load succeeds (hang loop responds to load only)

        var sw = System.Diagnostics.Stopwatch.StartNew();
        await Assert.ThrowsAsync<TimeoutException>(() => engine.PredictAsync("x"));
        sw.Stop();

        // Must return well under a hang (a real hang would exceed the test framework's default).
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(10),
            $"Predict should time out quickly, took {sw.Elapsed}.");
    }

    // H2: a worker that never even responds to load causes a startup TimeoutException.
    [Fact]
    public async Task SilentWorker_Load_ThrowsTimeoutWithinDeadline()
    {
        await using var engine = new TestOopEngine(silent: true, startupTimeout: TimeSpan.FromMilliseconds(300));

        await Assert.ThrowsAsync<TimeoutException>(() => engine.LoadAsync());
    }

    // M6: disposing during an in-flight predict waits for the in-flight call to resolve (here, a
    // timeout) instead of racing the pipe and surfacing an unexpected ObjectDisposedException.
    [Fact]
    public async Task Dispose_DuringInFlightPredict_WaitsAndDoesNotRace()
    {
        await using var engine = new TestOopEngine(hang: true, inferenceTimeout: TimeSpan.FromMilliseconds(400));
        await engine.LoadAsync();

        // Kick off a predict that will time out against the hung worker.
        Task<string> predict = engine.PredictAsync("x");
        await Task.Delay(50); // let it enter the in-flight read

        // Dispose concurrently; it must block on the in-flight lock until the predict times out.
        engine.Dispose();

        // The in-flight predict surfaces its TimeoutException (not an ObjectDisposedException),
        // proving Dispose waited rather than tearing the pipe out from under it.
        await Assert.ThrowsAsync<TimeoutException>(() => predict);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => engine.PredictAsync("after-dispose"));
    }

    // --- Test doubles --------------------------------------------------------------

    private sealed class TestOopEngine : OutOfProcessBaseInferenceEngine<string, string>, IAsyncDisposable
    {
        public int SpawnCount;
        private readonly bool _hang;
        private readonly bool _silent;
        private readonly TimeSpan _startupTimeout;
        private readonly TimeSpan _inferenceTimeout;
        private Task? _workerLoop;

        public bool IsWorkerLoaded => typeof(OutOfProcessBaseInferenceEngine<string, string>)
            .GetField("_workerLoaded", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!
            .GetValue(this) is true;

        public TestOopEngine(bool hang = false, bool silent = false,
            TimeSpan? startupTimeout = null, TimeSpan? inferenceTimeout = null)
            : base(new OutOfProcessEngineOptions
            {
                WorkerExecutablePath = "/dummy/worker",
                EngineId = "test",
                StartupTimeout = startupTimeout ?? TimeSpan.FromSeconds(10),
                InferenceTimeout = inferenceTimeout ?? TimeSpan.FromSeconds(10),
                EngineOptions = new InferenceEngineOptions { UseGpuAcceleration = false, WarmupOnLoad = false },
            })
        {
            _hang = hang;
            _silent = silent;
            _startupTimeout = startupTimeout ?? TimeSpan.FromSeconds(10);
            _inferenceTimeout = inferenceTimeout ?? TimeSpan.FromSeconds(10);
        }

        protected override void StartWorker()
        {
            Interlocked.Increment(ref SpawnCount);
            var (host, worker) = ChannelStream.CreatePair();
            // The host uses one duplex stream for both write (to worker) and read (from worker).
            SetWorkerStreamsForTesting(host, host);

            _workerLoop = _silent ? SilentWorkerLoop(worker)
                : _hang ? HangWorkerLoop(worker)
                : EchoWorkerLoop(worker);
        }

        private static async Task EchoWorkerLoop(ChannelStream worker)
        {
            // The real worker message loop drives an EchoEngine over the in-memory pipe.
            await InferenceWorkerHost.RunAsync<string, string>(
                _ => new EchoEngine(),
                new SystemTextJsonWorkerSerializer<string, string>(),
                input: worker,
                output: worker).ConfigureAwait(false);
        }

        private static async Task HangWorkerLoop(ChannelStream worker)
        {
            // Respond to load, then accept predict but never respond; block on a second read until
            // the host times out and disposes the pipe (unblocking this read with EOF).
            var loaded = await WorkerFrame.ReadAsync(worker, CancellationToken.None).ConfigureAwait(false);
            if (loaded is null) return;
            await WorkerFrame.WriteAsync(worker,
                new WorkerEnvelope { Kind = "loaded", EngineInfo = InferenceEngineInfo.Empty },
                CancellationToken.None).ConfigureAwait(false);

            var predict = await WorkerFrame.ReadAsync(worker, CancellationToken.None).ConfigureAwait(false);
            if (predict is null) return;
            // Hang here: the next read never completes until the host kills/disposes the pipe.
            await WorkerFrame.ReadAsync(worker, CancellationToken.None).ConfigureAwait(false);
        }

        private static async Task SilentWorkerLoop(ChannelStream worker)
        {
            // Never respond to anything; block on the first read until the host times out/disposes.
            await WorkerFrame.ReadAsync(worker, CancellationToken.None).ConfigureAwait(false);
        }

        public async ValueTask DisposeAsync()
        {
            Dispose();
            if (_workerLoop is not null)
            {
                try { await _workerLoop.ConfigureAwait(false); }
                catch { /* best-effort: worker loop teardown noise */ }
            }
        }
    }

    // A trivial engine the in-memory worker uses: echoes "echo:{input}".
    private sealed class EchoEngine : BaseInferenceEngine<string, string>
    {
        public EchoEngine() : base(new InferenceEngineOptions { UseGpuAcceleration = false, WarmupOnLoad = false }) { }

        public override Task LoadAsync() => Task.CompletedTask;
        protected override string ModelName => "echo";
        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => (new List<NamedOnnxValue>(), null);
        protected override string PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context) => string.Empty;
        public override Task<string> PredictAsync(string input) => Task.FromResult($"echo:{input}");
    }

    /// <summary>
    /// Minimal in-process duplex stream backed by two unbounded channels. Whatever one endpoint
    /// writes, the peer reads. Closing/disposing completes the writer so the peer's read sees EOF.
    /// Stands in for a redirected stdin/stdout pipe without spawning a process.
    /// </summary>
    private sealed class ChannelStream : Stream
    {
        private readonly ChannelReader<byte[]> _reader;
        private readonly ChannelWriter<byte[]> _writer;
        private byte[]? _pending;
        private int _pendingOffset;

        private ChannelStream(Channel<byte[]> readCh, Channel<byte[]> writeCh)
        {
            _reader = readCh.Reader;
            _writer = writeCh.Writer;
        }

        public static (ChannelStream a, ChannelStream b) CreatePair()
        {
            var c1 = Channel.CreateUnbounded<byte[]>();
            var c2 = Channel.CreateUnbounded<byte[]>();
            return (new ChannelStream(c1, c2), new ChannelStream(c2, c1));
        }

        public override bool CanRead => true;
        public override bool CanWrite => true;
        public override bool CanSeek => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override Task FlushAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            if (_pending is null || _pendingOffset >= _pending.Length)
            {
                if (!await _reader.WaitToReadAsync(cancellationToken).ConfigureAwait(false))
                    return 0; // channel completed + empty => clean EOF
                if (!_reader.TryRead(out var chunk))
                    return 0;
                _pending = chunk;
                _pendingOffset = 0;
            }
            int n = Math.Min(buffer.Length, _pending.Length - _pendingOffset);
            _pending.AsSpan(_pendingOffset, n).CopyTo(buffer.Span);
            _pendingOffset += n;
            return n;
        }

        public override ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
        {
            _writer.TryWrite(buffer.ToArray());
            return ValueTask.CompletedTask;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) _writer.TryComplete();
            base.Dispose(disposing);
        }
    }
}
