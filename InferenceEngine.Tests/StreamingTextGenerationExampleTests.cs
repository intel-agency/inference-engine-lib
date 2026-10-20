using InferenceEngine.Core;
using InferenceEngine.Examples;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace InferenceEngine.Tests;

/// <summary>
/// W4: the owner requires Examples counted in whole-codebase coverage, and
/// <see cref="StreamingTextGenerationExample"/> — a reference skeleton that ships no model and
/// no tokenizer — sat at 0%. These tests exercise its real contract: construction, the
/// hermetic model-resolution failure (its model is deliberately not in the registry), the
/// not-implemented guard on both streaming hooks (prefill and decode), and the one concrete
/// hook (<see cref="StreamingTextGenerationExample"/> concatenates step chunks) via an
/// exposing subclass. No production refactor was needed: the guards ARE the documented
/// behavior until a real HF-optimum export + tokenizer is supplied.
/// </summary>
public sealed class StreamingTextGenerationExampleTests : IDisposable
{
    private readonly List<StreamingTextGenerationExample> _engines = [];

    public void Dispose()
    {
        foreach (var engine in _engines)
            engine.Dispose();
    }

    [Fact]
    public async Task LoadAsync_ModelNotInRegistryAndNoUrls_FailsWithModelResolutionError()
    {
        var engine = CreateEngine();

        // streaming-text-generation.onnx is intentionally unregistered and provides no
        // download URLs — the load must fail hermetically before any session/network work.
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => engine.LoadAsync());

        Assert.Contains("streaming-text-generation.onnx", ex.Message);
    }

    [Theory]
    [InlineData(0)]   // prefill
    [InlineData(3)]   // decode with carried state
    public void BuildStepInputs_ThrowsNotSupportedException(int step)
    {
        var engine = CreateEngine();

        var ex = Assert.Throws<NotSupportedException>(
            () => engine.InvokeBuildStepInputs("prompt", state: null, step));

        Assert.Contains("Reference skeleton", ex.Message);
    }

    [Fact]
    public void ProcessStepOutputs_ThrowsNotSupportedException()
    {
        var engine = CreateEngine();

        var ex = Assert.Throws<NotSupportedException>(
            () => engine.InvokeProcessStepOutputs(outputs: null!, buildContext: null));

        Assert.Contains("Reference skeleton", ex.Message);
    }

    [Fact]
    public void BuildFinalOutput_ConcatenatesStepChunks()
    {
        var engine = CreateEngine();

        var result = engine.InvokeBuildFinalOutput(["Hel", "lo ", "world"]);

        Assert.Equal("Hello world", result);
    }

    private ExposedStreamingExample CreateEngine() 
    {
        var engine = new ExposedStreamingExample(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
        });
        _engines.Add(engine);
        return engine;
    }

    /// <summary>Exposes the protected streaming hooks so tests can drive them directly.</summary>
    private sealed class ExposedStreamingExample : StreamingTextGenerationExample
    {
        public ExposedStreamingExample(InferenceEngineOptions options)
            : base(options) { }

        public List<Microsoft.ML.OnnxRuntime.NamedOnnxValue> InvokeBuildStepInputs(string input, object? state, int step)
            => BuildStepInputs(input, state, step).Inputs;

        public (string Step, object? NextState, bool IsEos) InvokeProcessStepOutputs(
            Microsoft.ML.OnnxRuntime.IDisposableReadOnlyCollection<Microsoft.ML.OnnxRuntime.DisposableNamedOnnxValue> outputs,
            object? buildContext)
            => ProcessStepOutputs(outputs, buildContext);

        public string InvokeBuildFinalOutput(IReadOnlyList<string> steps) => BuildFinalOutput(steps);
    }
}
