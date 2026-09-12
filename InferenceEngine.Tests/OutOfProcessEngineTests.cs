using InferenceEngine.Core;
using InferenceEngine.Core.Diagnostics;
using InferenceEngine.Core.Worker;
using Microsoft.ML.OnnxRuntime;
using Xunit;

namespace InferenceEngine.Tests;

public sealed class OutOfProcessEngineTests
{
    [Fact]
    public async Task Framing_RoundTripsEnvelope()
    {
        using var ms = new MemoryStream();
        var original = new WorkerEnvelope
        {
            Kind = "loaded",
            EngineId = "x",
            Payload = "hello",
            EngineInfo = InferenceEngineInfo.Empty,
        };
        await WorkerFrame.WriteAsync(ms, original, CancellationToken.None);
        ms.Position = 0;

        var read = await WorkerFrame.ReadAsync(ms, CancellationToken.None);
        Assert.NotNull(read);
        Assert.Equal("loaded", read!.Kind);
        Assert.Equal("x", read.EngineId);
        Assert.Equal("hello", read.Payload);
        Assert.NotNull(read.EngineInfo);

        // A second read on the exhausted stream is a clean EOF (null).
        var second = await WorkerFrame.ReadAsync(ms, CancellationToken.None);
        Assert.Null(second);
    }

    // M7: a corrupt/oversized length prefix (256 MiB) is rejected before allocating a huge buffer.
    [Fact]
    public async Task Framing_RejectsOversizedLengthPrefix_WithoutHugeAllocation()
    {
        using var ms = new MemoryStream();
        // Corrupt prefix claiming 256 MiB — far above the 16 MiB cap. No payload bytes follow.
        ms.Write(BitConverter.GetBytes(256 * 1024 * 1024), 0, 4);
        ms.Position = 0;

        await Assert.ThrowsAsync<InvalidDataException>(
            () => WorkerFrame.ReadAsync(ms, CancellationToken.None));
    }

    [Fact]
    public void Serializer_RoundTripsByteArrayAndString()
    {
        var serializer = new SystemTextJsonWorkerSerializer<byte[], string>();
        var bytes = new byte[] { 1, 2, 3 };
        var serializedInput = serializer.SerializeInput(bytes);
        Assert.Equal(bytes, serializer.DeserializeInput(serializedInput));

        var serializedOutput = serializer.SerializeOutput("hi");
        Assert.Equal("hi", serializer.DeserializeOutput(serializedOutput));
    }

    [Fact]
    public async Task WorkerHost_LoadThenPredictThenShutdown()
    {
        var serializer = new SystemTextJsonWorkerSerializer<string, string>();

        using var requests = new MemoryStream();
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope
        {
            Kind = "load",
            EngineId = "e1",
            Options = new WorkerEngineOptions { ModelPath = "/m.onnx" },
        }, CancellationToken.None);
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope
        {
            Kind = "predict",
            Payload = serializer.SerializeInput("world"),
        }, CancellationToken.None);
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope { Kind = "shutdown" }, CancellationToken.None);
        requests.Position = 0;

        using var responses = new MemoryStream();

        var run = Task.Run(() => InferenceWorkerHost.RunAsync<string, string>(
            _ => new FakeWorkerEngine(),
            serializer,
            input: requests,
            output: responses));

        if (await Task.WhenAny(run, Task.Delay(TimeSpan.FromSeconds(10))) != run)
        {
            Assert.Fail("InferenceWorkerHost.RunAsync did not complete within the timeout.");
        }
        var exitCode = await run;
        Assert.Equal(0, exitCode);

        responses.Position = 0;
        var loaded = await WorkerFrame.ReadAsync(responses, CancellationToken.None);
        Assert.NotNull(loaded);
        Assert.Equal("loaded", loaded!.Kind);
        Assert.NotNull(loaded.EngineInfo);
        Assert.Null(loaded.Error);

        var result = await WorkerFrame.ReadAsync(responses, CancellationToken.None);
        Assert.NotNull(result);
        Assert.Equal("result", result!.Kind);
        Assert.Null(result.Error);
        Assert.NotNull(result.Payload);
        Assert.Equal("echo:world", serializer.DeserializeOutput(result.Payload!));

        var eof = await WorkerFrame.ReadAsync(responses, CancellationToken.None);
        Assert.Null(eof);
    }

    [Fact]
    public async Task WorkerHost_LoadError_IsReported()
    {
        using var requests = new MemoryStream();
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope
        {
            Kind = "load",
            EngineId = "e1",
            Options = new WorkerEngineOptions { ModelPath = "/missing.onnx" },
        }, CancellationToken.None);
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope { Kind = "shutdown" }, CancellationToken.None);
        requests.Position = 0;

        using var responses = new MemoryStream();

        var run = Task.Run(() => InferenceWorkerHost.RunAsync<string, string>(
            _ => new ThrowingLoadEngine(),
            input: requests,
            output: responses));

        if (await Task.WhenAny(run, Task.Delay(TimeSpan.FromSeconds(10))) != run)
        {
            Assert.Fail("InferenceWorkerHost.RunAsync did not complete within the timeout.");
        }
        await run;

        responses.Position = 0;
        var loaded = await WorkerFrame.ReadAsync(responses, CancellationToken.None);
        Assert.NotNull(loaded);
        Assert.Equal("loaded", loaded!.Kind);
        Assert.NotNull(loaded.Error);
        Assert.Contains(LoadExceptionMessage, loaded.Error);
    }

    [Fact]
    public async Task WorkerHost_PredictBeforeLoad_RepliesWithError()
    {
        using var requests = new MemoryStream();
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope
        {
            Kind = "predict",
            Payload = "\"world\"",
        }, CancellationToken.None);
        await WorkerFrame.WriteAsync(requests, new WorkerEnvelope { Kind = "shutdown" }, CancellationToken.None);
        requests.Position = 0;

        using var responses = new MemoryStream();

        var run = Task.Run(() => InferenceWorkerHost.RunAsync<string, string>(
            _ => new FakeWorkerEngine(),
            input: requests,
            output: responses));

        if (await Task.WhenAny(run, Task.Delay(TimeSpan.FromSeconds(10))) != run)
        {
            Assert.Fail("InferenceWorkerHost.RunAsync did not complete within the timeout.");
        }
        await run;

        responses.Position = 0;
        var result = await WorkerFrame.ReadAsync(responses, CancellationToken.None);
        Assert.NotNull(result);
        Assert.Equal("result", result!.Kind);
        Assert.NotNull(result.Error);
    }

    [Fact]
    public async Task Proxy_LoadErrorFrame_StatusFailedWithPartialEngineInfo()
    {
        // Worker answers "load" with an error frame plus the partial EngineInfo it had built
        // (provider selection ran, session construction failed). The proxy's status must carry
        // both, mirrored onto the shared object model.
        using var toWorker = new MemoryStream();
        using var fromWorker = new MemoryStream();
        var partialInfo = new InferenceEngineInfo
        {
            Provider = InferenceProvider.Cpu,
            IsFallback = true,
            FallbackReason = InferenceFallbackReason.HsaRuntimeUnloadable,
            Checks = new[] { ProviderDiagnosticCheck.Fail("hsa.dlopen", "libhsa-runtime64.so is loadable", "Provider", 0, "dlerror") },
        };
        await WorkerFrame.WriteAsync(fromWorker, new WorkerEnvelope
        {
            Kind = "loaded",
            EngineId = "e1",
            EngineInfo = partialInfo,
            Error = LoadExceptionMessage,
        }, CancellationToken.None);
        fromWorker.Position = 0;

        var proxy = new NoSpawnProxy(MakeOptions("e1"));
        proxy.SetWorkerStreamsForTesting(toWorker, fromWorker);

        await Assert.ThrowsAsync<InvalidOperationException>(() => proxy.LoadAsync());

        Assert.Equal(EngineLoadState.Failed, proxy.Status.State);
        Assert.Equal(LoadExceptionMessage, proxy.Status.LoadError);
        Assert.NotNull(proxy.Status.Info);
        Assert.Equal(InferenceFallbackReason.HsaRuntimeUnloadable, proxy.Status.Info!.FallbackReason);
        Assert.Single(proxy.Status.Info.Checks);
    }

    [Fact]
    public async Task Proxy_PipeEofDuringLoad_StatusFailedWithExceptionMessage()
    {
        using var toWorker = new MemoryStream();
        using var fromWorker = new MemoryStream(); // empty: worker "died" before responding

        var proxy = new NoSpawnProxy(MakeOptions("e1"));
        proxy.SetWorkerStreamsForTesting(toWorker, fromWorker);

        await Assert.ThrowsAsync<InvalidOperationException>(() => proxy.LoadAsync());

        Assert.Equal(EngineLoadState.Failed, proxy.Status.State);
        Assert.Null(proxy.Status.Info);
        Assert.Contains("closed the pipe", proxy.Status.LoadError);
    }

    [Fact]
    public async Task Proxy_SuccessfulLoad_StatusLoadedWithMirroredInfo()
    {
        using var toWorker = new MemoryStream();
        using var fromWorker = new MemoryStream();
        await WorkerFrame.WriteAsync(fromWorker, new WorkerEnvelope
        {
            Kind = "loaded",
            EngineId = "e1",
            EngineInfo = new InferenceEngineInfo { Provider = InferenceProvider.Cpu },
        }, CancellationToken.None);
        fromWorker.Position = 0;

        var proxy = new NoSpawnProxy(MakeOptions("e1"));
        proxy.SetWorkerStreamsForTesting(toWorker, fromWorker);

        await proxy.LoadAsync();

        Assert.Equal(EngineLoadState.Loaded, proxy.Status.State);
        Assert.Equal(InferenceProvider.Cpu, proxy.Status.Info?.Provider);
        Assert.Null(proxy.Status.LoadError);
        Assert.False(proxy.Status.IsGpuActive);
    }

    [Fact]
    public async Task Proxy_LoadStatusChanges_FiredThroughCallback()
    {
        using var toWorker = new MemoryStream();
        using var fromWorker = new MemoryStream();
        await WorkerFrame.WriteAsync(fromWorker, new WorkerEnvelope
        {
            Kind = "loaded",
            EngineId = "e1",
            EngineInfo = new InferenceEngineInfo { Provider = InferenceProvider.MIGraphX, DeviceId = 0 },
        }, CancellationToken.None);
        fromWorker.Position = 0;

        var options = MakeOptions("e1");
        var observed = new List<InferenceEngineStatus>();
        options.EngineOptions.OnStatusChanged = s => observed.Add(s);
        var proxy = new NoSpawnProxy(options);
        proxy.SetWorkerStreamsForTesting(toWorker, fromWorker);

        await proxy.LoadAsync();

        Assert.Equal(new[] { EngineLoadState.Loading, EngineLoadState.Loaded },
            observed.Select(s => s.State).ToArray());
        Assert.True(observed[^1].IsGpuActive);
    }

    private static OutOfProcessEngineOptions MakeOptions(string engineId) => new()
    {
        WorkerExecutablePath = "unused",
        WorkerArguments = ["--worker", engineId],
        EngineId = engineId,
        StartupTimeout = TimeSpan.FromSeconds(5),
    };

    // Proxy with process spawning disabled: streams are injected directly.
    private sealed class NoSpawnProxy(OutOfProcessEngineOptions options)
        : OutOfProcessBaseInferenceEngine<string, string>(options)
    {
        protected override void StartWorker() { /* no process; streams injected by the test */ }
    }

    private const string LoadExceptionMessage = "boom: load failed";

    // A minimal BaseInferenceEngine subclass that does NOT load a real ONNX session.
    private sealed class FakeWorkerEngine : BaseInferenceEngine<string, string>
    {
        public FakeWorkerEngine() : base(new InferenceEngineOptions()) { }

        public override Task LoadAsync() => Task.CompletedTask;

        protected override string ModelName => "fake";

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => (new List<NamedOnnxValue>(), null);

        protected override string PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context) => "";

        public override Task<string> PredictAsync(string input) => Task.FromResult($"echo:{input}");
    }

    // Engine whose LoadAsync throws, to exercise the worker's error reporting path.
    private sealed class ThrowingLoadEngine : BaseInferenceEngine<string, string>
    {
        public ThrowingLoadEngine() : base(new InferenceEngineOptions()) { }

        public override Task LoadAsync() => throw new InvalidOperationException(LoadExceptionMessage);

        protected override string ModelName => "throwing";

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => (new List<NamedOnnxValue>(), null);

        protected override string PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context) => "";

        public override Task<string> PredictAsync(string input) => Task.FromResult($"echo:{input}");
    }
}
