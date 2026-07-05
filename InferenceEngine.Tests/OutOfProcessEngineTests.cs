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
