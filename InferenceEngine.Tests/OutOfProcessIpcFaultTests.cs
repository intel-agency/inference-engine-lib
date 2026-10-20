using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using InferenceEngine.Core;
using InferenceEngine.Core.Diagnostics;
using InferenceEngine.Core.Worker;
using Microsoft.ML.OnnxRuntime;
using Xunit;


namespace InferenceEngine.Tests;

/// <summary>
/// Fault-injecting coverage for the out-of-process IPC stack: worker crash/EOF, write/read
/// timeouts, malformed and oversized frames, missing-payload responses, worker-side factory /
/// predict failures, loop cancellation, and real-subprocess spawn/kill/teardown paths that the
/// happy-path tests never exercise. Two harnesses are used: an in-proc fake worker speaking the
/// stdin/stdout framing contract over preloaded <see cref="MemoryStream"/>s or a
/// <see cref="ChannelStream"/> pair, and real tiny <c>/bin/sh</c> subprocesses that die, hang, or
/// echo garbage (Linux only; skipped elsewhere).
/// </summary>
public sealed class OutOfProcessIpcFaultTests
{
    // --- proxy: load/predict fault frames over preloaded in-memory pipes -------------------

    // Worker answers "loaded" with neither Error nor EngineInfo → proxy must fail with the
    // dedicated no-EngineInfo error and a Failed status.
    [Fact]
    public async Task Proxy_LoadResponseWithoutEngineInfo_Fails()
    {
        using var proxy = NewPreloadedProxy(
            Frame(new WorkerEnvelope { Kind = "loaded", EngineId = "e1" }));

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => proxy.LoadAsync());

        Assert.Contains("no EngineInfo", ex.Message);
        Assert.Equal(EngineLoadState.Failed, proxy.Status.State);
        Assert.Equal("Worker load response had no EngineInfo.", proxy.Status.LoadError);
    }

    // Worker answers "predict" with an Error frame → the proxy surfaces the worker-side error.
    [Fact]
    public async Task Proxy_PredictErrorFrame_ThrowsWorkerError()
    {
        using var proxy = NewPreloadedProxy(
            Frame(new WorkerEnvelope { Kind = "loaded", EngineInfo = InferenceEngineInfo.Empty }),
            Frame(new WorkerEnvelope { Kind = "result", Error = "predict boom" }));
        await proxy.LoadAsync();

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => proxy.PredictAsync("x"));

        Assert.Contains("predict boom", ex.Message);
    }

    // Worker answers "predict" with a result frame that carries no payload.
    [Fact]
    public async Task Proxy_PredictResultWithoutPayload_Throws()
    {
        using var proxy = NewPreloadedProxy(
            Frame(new WorkerEnvelope { Kind = "loaded", EngineInfo = InferenceEngineInfo.Empty }),
            Frame(new WorkerEnvelope { Kind = "result" }));
        await proxy.LoadAsync();

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => proxy.PredictAsync("x"));

        Assert.Contains("no payload", ex.Message);
    }

    // Worker closes the pipe between load and predict → predict sees EOF.
    [Fact]
    public async Task Proxy_PredictPipeEof_Throws()
    {
        using var proxy = NewPreloadedProxy(
            Frame(new WorkerEnvelope { Kind = "loaded", EngineInfo = InferenceEngineInfo.Empty }));
        await proxy.LoadAsync();

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => proxy.PredictAsync("x"));

        Assert.Contains("closed the pipe during predict", ex.Message);
    }

    // Predict without a prior LoadAsync auto-loads the worker first (in-flight auto-load path).
    [Fact]
    public async Task Proxy_PredictWithoutLoad_AutoLoadsAndSucceeds()
    {
        using var proxy = NewPreloadedProxy(
            Frame(new WorkerEnvelope { Kind = "loaded", EngineInfo = InferenceEngineInfo.Empty }),
            Frame(new WorkerEnvelope { Kind = "result", Payload = "\"hello\"" }));

        Assert.Equal("hello", await proxy.PredictAsync("x"));
    }

    // Predict after Dispose must throw ObjectDisposedException, not touch the torn-down pipe.
    [Fact]
    public async Task Proxy_PredictAfterDispose_ThrowsObjectDisposed()
    {
        using var proxy = NewPreloadedProxy(
            Frame(new WorkerEnvelope { Kind = "loaded", EngineInfo = InferenceEngineInfo.Empty }));
        await proxy.LoadAsync();
        proxy.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(() => proxy.PredictAsync("x"));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => proxy.LoadAsync());
    }

    // A second LoadAsync after a failed one retries (StartWorker cleanup + respawn).
    [Fact]
    public async Task Proxy_LoadRetryAfterFailure_TriesAgain()
    {
        using var proxy = NewPreloadedProxy(
            Frame(new WorkerEnvelope { Kind = "loaded", EngineId = "e1" }), // no EngineInfo → fail
            Frame(new WorkerEnvelope { Kind = "loaded", EngineInfo = InferenceEngineInfo.Empty }));

        await Assert.ThrowsAsync<InvalidOperationException>(() => proxy.LoadAsync());
        await proxy.LoadAsync(); // retry succeeds against the second preloaded frame

        Assert.Equal(EngineLoadState.Loaded, proxy.Status.State);
    }

    // Worker emits a malformed (non-JSON) frame during load → JsonException propagates and the
    // status transitions to Failed with the parser error.
    [Fact]
    public async Task Proxy_MalformedFrameDuringLoad_StatusFailed()
    {
        var fromWorker = new MemoryStream();
        fromWorker.Write(BitConverter.GetBytes(13), 0, 4);
        fromWorker.Write(System.Text.Encoding.UTF8.GetBytes("not-json-at-all"), 0, 13);
        fromWorker.Position = 0;
        using var proxy = new NoSpawnProxy(MakeOptions());
        proxy.SetWorkerStreamsForTesting(new MemoryStream(), fromWorker);

        await Assert.ThrowsAsync<JsonException>(() => proxy.LoadAsync());

        Assert.Equal(EngineLoadState.Failed, proxy.Status.State);
        Assert.NotNull(proxy.Status.LoadError);
    }

    [Fact]
    public void Proxy_ModelName_IsEngineId()
    {
        using var proxy = new NoSpawnProxy(MakeOptions());
        Assert.Equal("e1", proxy.ModelNameForTest);
    }

    // --- proxy: write/read timeouts over injected streams -----------------------------------

    // A worker whose stdin never accepts the frame → write timeout kills the (fake) pipe and
    // surfaces TimeoutException instead of hanging forever.
    [Fact]
    public async Task Proxy_WriteTimeout_ThrowsAndFailsStatus()
    {
        using var blocker = new BlockUntilDisposedStream();
        using var proxy = new NoSpawnProxy(MakeOptions(startupTimeout: TimeSpan.FromMilliseconds(200)));
        proxy.SetWorkerStreamsForTesting(blocker, new MemoryStream());

        var ex = await Assert.ThrowsAsync<TimeoutException>(() => proxy.LoadAsync());

        Assert.Contains("did not accept a write", ex.Message);
        Assert.Equal(EngineLoadState.Failed, proxy.Status.State);
    }

    // --- WorkerFrame: framing corruption ------------------------------------------------------

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public async Task Framing_RejectsNonPositiveLengthPrefix(int bogusLength)
    {
        using var ms = new MemoryStream();
        ms.Write(BitConverter.GetBytes(bogusLength), 0, 4);
        ms.Position = 0;

        await Assert.ThrowsAsync<InvalidDataException>(() => WorkerFrame.ReadAsync(ms, CancellationToken.None));
    }

    // Length prefix promises a payload but the stream ends before/inside it →
    // EndOfStreamException, not a partial/failed parse. Zero following bytes hits the EOF-at-
    // payload-start check; ten of a promised 100 hits the mid-frame EOF inside the exact reader.
    [Theory]
    [InlineData(0)]
    [InlineData(10)]
    public async Task Framing_TruncatedPayload_ThrowsEndOfStream(int bytesFollow)
    {
        using var ms = new MemoryStream();
        ms.Write(BitConverter.GetBytes(100), 0, 4);
        ms.Write(new byte[bytesFollow], 0, bytesFollow);
        ms.Position = 0;

        await Assert.ThrowsAsync<EndOfStreamException>(() => WorkerFrame.ReadAsync(ms, CancellationToken.None));
    }

    // A frame of a legal size whose payload is not JSON → JsonException from the deserialize.
    [Fact]
    public async Task Framing_MalformedJsonPayload_ThrowsJsonException()
    {
        var garbage = System.Text.Encoding.UTF8.GetBytes("<<<not json>>>");
        using var ms = new MemoryStream();
        ms.Write(BitConverter.GetBytes(garbage.Length), 0, 4);
        ms.Write(garbage, 0, garbage.Length);
        ms.Position = 0;

        await Assert.ThrowsAsync<JsonException>(() => WorkerFrame.ReadAsync(ms, CancellationToken.None));
    }

    // --- serializer: null round-trip branch ---------------------------------------------------

    [Fact]
    public void Serializer_NullPayload_ThrowsInvalidOperationException()
    {
        var serializer = new SystemTextJsonWorkerSerializer<string, string>();

        Assert.Throws<InvalidOperationException>(() => serializer.DeserializeInput("null"));
        Assert.Throws<InvalidOperationException>(() => serializer.DeserializeOutput("null"));
    }

    // --- worker host: loop fault paths --------------------------------------------------------

    // Cancellation observed between requests → clean exit 0 (OperationCanceledException path).
    [Fact]
    public async Task WorkerHost_CancellationDuringRead_ExitsCleanly()
    {
        var (requests, peer) = ChannelStream.CreatePair();
        using var responses = new MemoryStream();
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));

        var run = InferenceWorkerHost.RunAsync<string, string>(
            _ => new EchoEngine(), input: requests, output: responses, cancellationToken: cts.Token);

        var exitCode = await run.WaitAsync(TimeSpan.FromSeconds(10));
        peer.Dispose();
        Assert.Equal(0, exitCode);
    }

    // Empty stdin (immediate EOF, no frames) → loop exits 0.
    [Fact]
    public async Task WorkerHost_StdinEof_ExitsCleanly()
    {
        using var requests = new MemoryStream();
        using var responses = new MemoryStream();

        var exitCode = await InferenceWorkerHost.RunAsync<string, string>(
            _ => new EchoEngine(), input: requests, output: responses);

        Assert.Equal(0, exitCode);
    }

    // Engine factory throws → "loaded" error frame, worker stays alive for the next command.
    [Fact]
    public async Task WorkerHost_FactoryThrows_ReportsErrorAndContinues()
    {
        using var requests = new MemoryStream();
        // Options deliberately null: the loop must fall back to default WorkerEngineOptions.
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope { Kind = "load", EngineId = "e1" },
            CancellationToken.None);
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope { Kind = "shutdown" }, CancellationToken.None);
        requests.Position = 0;
        using var responses = new MemoryStream();

        await InferenceWorkerHost.RunAsync<string, string>(
            _ => throw new InvalidOperationException("factory boom"),
            input: requests, output: responses);

        responses.Position = 0;
        var loaded = await WorkerFrame.ReadAsync(responses, CancellationToken.None);
        Assert.NotNull(loaded);
        Assert.Contains("factory boom", loaded!.Error);
    }

    // Engine predict throws → "result" error frame.
    [Fact]
    public async Task WorkerHost_PredictThrows_ReportsErrorFrame()
    {
        using var requests = new MemoryStream();
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope
        {
            Kind = "load",
            Options = new WorkerEngineOptions { ModelPath = "/m.onnx" },
        }, CancellationToken.None);
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope { Kind = "predict", Payload = "\"x\"" },
            CancellationToken.None);
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope { Kind = "shutdown" }, CancellationToken.None);
        requests.Position = 0;
        using var responses = new MemoryStream();

        await InferenceWorkerHost.RunAsync<string, string>(
            _ => new ThrowingPredictEngine(), input: requests, output: responses);

        responses.Position = 0;
        var loaded = await WorkerFrame.ReadAsync(responses, CancellationToken.None);
        Assert.NotNull(loaded);
        Assert.Null(loaded!.Error);
        var result = await WorkerFrame.ReadAsync(responses, CancellationToken.None);
        Assert.NotNull(result);
        Assert.Contains("predict boom", result!.Error);
    }

    // Unknown message kind → error result frame (dispatch error path).
    [Fact]
    public async Task WorkerHost_UnknownKind_RepliesWithError()
    {
        using var requests = new MemoryStream();
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope { Kind = "bogus" }, CancellationToken.None);
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope { Kind = "shutdown" }, CancellationToken.None);
        requests.Position = 0;
        using var responses = new MemoryStream();

        await InferenceWorkerHost.RunAsync<string, string>(
            _ => new EchoEngine(), input: requests, output: responses);

        responses.Position = 0;
        var result = await WorkerFrame.ReadAsync(responses, CancellationToken.None);
        Assert.NotNull(result);
        Assert.Contains("Unknown or unexpected message kind", result!.Error);
    }

    // Production mode (no injected output): Console.Out is redirected to stderr so stray writes
    // cannot corrupt the protocol channel; a shutdown frame then exits 0. Console.Out is restored.
    [Fact]
    public async Task WorkerHost_ConsoleMode_RedirectsConsoleOut()
    {
        var savedOut = Console.Out;
        try
        {
            using var requests = new MemoryStream();
            await WorkerFrame.WriteAsync(requests, new WorkerEnvelope { Kind = "shutdown" },
                CancellationToken.None);
            requests.Position = 0;

            var exitCode = await InferenceWorkerHost.RunAsync<string, string>(
                _ => new EchoEngine(), input: requests, output: null);

            Assert.Equal(0, exitCode);
        }
        finally
        {
            Console.SetOut(savedOut);
        }
    }

    // --- real subprocesses (Linux): spawn / kill / teardown / stderr pump ---------------------

    // Worker dies before answering load → EOF surfaces as a load failure; a retry respawns
    // (exercising the StartWorker cleanup of a previously-started process).
    [SkippableFact]
    public async Task RealProcess_WorkerDiesAtSpawn_LoadFailsThenRetryRespawns()
    {
        Skip.IfNot(OperatingSystem.IsLinux(), "/bin/sh subprocess harness is Linux-only");

        await using var proxy = new RealWorkerProxy(MakeOptions(), "/bin/true", env: new Dictionary<string, string>());

        await Assert.ThrowsAsync<InvalidOperationException>(() => proxy.LoadAsync());
        Assert.Equal(EngineLoadState.Failed, proxy.Status.State);

        // Retry: StartWorker must clean up the dead process and spawn a fresh one.
        await Assert.ThrowsAsync<InvalidOperationException>(() => proxy.LoadAsync());
        Assert.Equal(2, proxy.SpawnCount);

        await proxy.WaitForExitHandlersAsync(); // let the forensic Exited handler run
    }

    // Worker hangs past the startup timeout → read timeout kills the real process.
    [SkippableFact]
    public async Task RealProcess_HungWorker_LoadTimesOutAndKills()
    {
        Skip.IfNot(OperatingSystem.IsLinux(), "/bin/sh subprocess harness is Linux-only");

        var options = MakeOptions(startupTimeout: TimeSpan.FromMilliseconds(500));
        await using var proxy = new RealWorkerProxy(options, "/bin/sh",
            args: ["-c", "exec sleep 30"], env: new Dictionary<string, string> { ["OOP_TEST_VAR"] = "1" });

        var ex = await Assert.ThrowsAsync<TimeoutException>(() => proxy.LoadAsync());

        Assert.Contains("did not respond", ex.Message);
        await proxy.WaitForExitHandlersAsync();
    }

    // A cat worker echoes the load frame back (no EngineInfo) → load fails while the process is
    // still ALIVE; the sync Dispose path then sends a graceful shutdown frame to a live process,
 // closes stdin first, and reaps it. Also: a retry kills the still-running previous worker, and
    // worker stderr lines flow through HandleWorkerStderrLine.
    [SkippableFact]
    public async Task RealProcess_EchoWorker_FailsLoadAndShutsDownGracefully()
    {
        Skip.IfNot(OperatingSystem.IsLinux(), "/bin/sh subprocess harness is Linux-only");

        var options = MakeOptions();
        var proxy = new StderrRecordingProxy(options, "/bin/sh",
            args: ["-c", "echo 'Model Compile: Begin' >&2; exec cat"],
            env: new Dictionary<string, string>());

        await Assert.ThrowsAsync<InvalidOperationException>(() => proxy.LoadAsync());
        Assert.Equal(EngineLoadState.Failed, proxy.Status.State);

        // Retry: the previous worker (cat) is still alive → the cleanup path must Kill it.
        await Assert.ThrowsAsync<InvalidOperationException>(() => proxy.LoadAsync());
        Assert.Equal(2, proxy.SpawnCount);

        // The worker's stderr marker must have flowed through HandleWorkerStderrLine.
        var sawMarker = await proxy.WaitForStderrLineAsync(l => l.Contains("Model Compile: Begin"),
            TimeSpan.FromSeconds(5));
        Assert.True(sawMarker, "the worker stderr marker never reached HandleWorkerStderrLine");

        // Async dispose against the LIVE process: graceful shutdown frame over the pipe, then
        // stdin-first teardown reaps the worker.
        await proxy.DisposeAsync();
        await proxy.WaitForExitHandlersAsync();
    }

    // --- helpers -------------------------------------------------------------------------------

    private static byte[] Frame(WorkerEnvelope envelope)
    {
        using var ms = new MemoryStream();
        WorkerFrame.Write(ms, envelope);
        return ms.ToArray();
    }

    private static NoSpawnProxy NewPreloadedProxy(params byte[][] frames)
    {
        var fromWorker = new MemoryStream(frames.SelectMany(b => b).ToArray());
        var proxy = new NoSpawnProxy(MakeOptions());
        proxy.SetWorkerStreamsForTesting(new MemoryStream(), fromWorker);
        return proxy;
    }

    private static OutOfProcessEngineOptions MakeOptions(TimeSpan? startupTimeout = null) => new()
    {
        WorkerExecutablePath = "unused",
        WorkerArguments = ["--worker", "inference"],
        EngineId = "e1",
        StartupTimeout = startupTimeout ?? TimeSpan.FromSeconds(10),
        InferenceTimeout = TimeSpan.FromSeconds(10),
        EngineOptions = new InferenceEngineOptions { UseGpuAcceleration = false, WarmupOnLoad = false },
    };

    // Proxy with process spawning disabled: streams are injected directly by the test.
    private sealed class NoSpawnProxy(OutOfProcessEngineOptions options)
        : OutOfProcessBaseInferenceEngine<string, string>(options)
    {
        public string ModelNameForTest => ModelName;

        protected override void StartWorker() { /* no process; streams injected by the test */ }
    }

    // Proxy that spawns a REAL tiny subprocess (/bin/true, /bin/sh -c …) so the process
    // spawn/cleanup/kill/stderr/Exited paths run against a genuine Process object.
    private class RealWorkerProxy(OutOfProcessEngineOptions options, string exe,
        IReadOnlyList<string>? args = null, IReadOnlyDictionary<string, string>? env = null)
        : OutOfProcessBaseInferenceEngine<string, string>(
            new OutOfProcessEngineOptions
            {
                WorkerExecutablePath = exe,
                WorkerArguments = (args as IReadOnlyList<string>) ?? Array.Empty<string>(),
                WorkerEnvironment = env ?? new Dictionary<string, string>(),
                EngineId = options.EngineId,
                StartupTimeout = options.StartupTimeout,
                InferenceTimeout = options.InferenceTimeout,
                EngineOptions = options.EngineOptions,
            })
    {
        public int SpawnCount;

        protected override void StartWorker()
        {
            Interlocked.Increment(ref SpawnCount);
            base.StartWorker();
        }

        // The Exited forensic handler runs on a threadpool thread; wait briefly so it runs before
        // the test ends.
        public Task WaitForExitHandlersAsync() => Task.Delay(300);
    }

    // RealWorkerProxy that records every stderr line that reached the handler.
    private sealed class StderrRecordingProxy(OutOfProcessEngineOptions options, string exe,
        IReadOnlyList<string> args, IReadOnlyDictionary<string, string> env)
        : RealWorkerProxy(options, exe, args, env)
    {
        private readonly List<string> _lines = new();
        private readonly object _gate = new();

        protected override void HandleWorkerStderrLine(string line)
        {
            lock (_gate) _lines.Add(line);
            base.HandleWorkerStderrLine(line);
        }

        public async Task<bool> WaitForStderrLineAsync(Func<string, bool> predicate, TimeSpan timeout)
        {
            var deadline = DateTime.UtcNow + timeout;
            while (DateTime.UtcNow < deadline)
            {
                lock (_gate)
                    if (_lines.Any(predicate))
                        return true;
                await Task.Delay(25);
            }
            return false;
        }
    }

    // Stream whose WriteAsync parks until the stream is disposed — a worker that never accepts
    // a frame on its stdin. KillWorker disposes the stream, which unblocks (and fails) the
    // abandoned write while the caller surfaces the TimeoutException.
    private sealed class BlockUntilDisposedStream : Stream
    {
        private readonly TaskCompletionSource _released = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public override bool CanRead => true;
        public override bool CanWrite => true;
        public override bool CanSeek => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override Task FlushAsync(CancellationToken ct) => Task.CompletedTask;
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken ct = default)
        {
            await Task.Delay(Timeout.Infinite, ct).ConfigureAwait(false);
            return 0; // unreachable: cancelled first
        }

        public override async ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken ct = default)
        {
            await _released.Task.ConfigureAwait(false);
            throw new ObjectDisposedException(nameof(BlockUntilDisposedStream));
        }

        protected override void Dispose(bool disposing)
        {
            _released.TrySetResult();
            base.Dispose(disposing);
        }
    }

    private sealed class EchoEngine : BaseInferenceEngine<string, string>
    {
        public EchoEngine() : base(new InferenceEngineOptions()) { }

        public override Task LoadAsync() => Task.CompletedTask;
        protected override string ModelName => "echo";
        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => (new List<NamedOnnxValue>(), null);
        protected override string PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context) => string.Empty;
        public override Task<string> PredictAsync(string input) => Task.FromResult($"echo:{input}");
    }

    private sealed class ThrowingPredictEngine : BaseInferenceEngine<string, string>
    {
        public ThrowingPredictEngine() : base(new InferenceEngineOptions()) { }

        public override Task LoadAsync() => Task.CompletedTask;
        protected override string ModelName => "throwing-predict";
        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => (new List<NamedOnnxValue>(), null);
        protected override string PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context) => string.Empty;
        public override Task<string> PredictAsync(string input) => throw new InvalidOperationException("predict boom");
    }
}
