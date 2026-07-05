using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using InferenceEngine.Core;
using InferenceEngine.Core.ModelProviders;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Xunit;

namespace InferenceEngine.Tests;

public class BaseInferenceEngineTests : IDisposable
{
    private readonly string _testDir;

    public BaseInferenceEngineTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"engine-tests-{Guid.NewGuid()}");
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            try
            {
                Directory.Delete(_testDir, true);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Suppress filesystem exceptions during cleanup to prevent escaping from Dispose (CA1065)
            }
        }
    }

    [Fact]
    public void Constructor_WithNullOptions_ShouldThrowArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new TestEngine(null!, NullLogger.Instance));
    }

    [Fact]
    public void Constructor_WithNullLogger_ShouldUseNullLogger()
    {
        var options = new InferenceEngineOptions { ModelPath = "test.onnx" };
        var engine = new TestEngine(options, null);
        Assert.NotNull(engine);
        engine.Dispose();
    }

    [Fact]
    public void ModelName_ShouldReturnExpectedValue()
    {
        var engine = CreateTestEngine();
        Assert.Equal("test.onnx", engine.GetModelName());
        engine.Dispose();
    }

    [Fact]
    public void ModelDownloadUrls_ShouldReturnExpectedValue()
    {
        var engine = CreateTestEngine();
        var urls = engine.GetModelDownloadUrls();
        Assert.Single(urls);
        Assert.Equal("http://example.com/test.onnx", urls[0]);
        engine.Dispose();
    }

    [Fact]
    public void ModelCacheDirectory_ShouldReturnDefaultPath()
    {
        var engine = CreateTestEngine();
        var cacheDir = engine.GetModelCacheDirectory();
        Assert.Contains("InferenceEngine", cacheDir);
        Assert.Contains("Models", cacheDir);
        engine.Dispose();
    }

    [Fact]
    public void Dispose_CalledMultipleTimes_ShouldNotThrow()
    {
        var engine = CreateTestEngine();
        engine.Dispose();
        engine.Dispose(); // Should not throw
    }

    [Fact]
    public void Dispose_AfterPartialInitialization_ShouldNotThrow()
    {
        var engine = CreateTestEngine();
        // Don't call LoadAsync, just dispose
        engine.Dispose();
    }

    [Fact]
    public void InputMetadata_BeforeLoad_ShouldBeNull()
    {
        var engine = CreateTestEngine();
        Assert.Null(engine.GetInputMetadata());
        engine.Dispose();
    }

    [Fact]
    public void OutputMetadata_BeforeLoad_ShouldBeNull()
    {
        var engine = CreateTestEngine();
        Assert.Null(engine.GetOutputMetadata());
        engine.Dispose();
    }

    [Fact]
    public void PrimaryInputName_BeforeLoad_ShouldBeNull()
    {
        var engine = CreateTestEngine();
        Assert.Null(engine.GetPrimaryInputName());
        engine.Dispose();
    }

    [Fact]
    public async Task LoadAsync_WithExplicitModelProvider_ShouldUseProvider()
    {
        var modelPath = Path.Combine(_testDir, "explicit.onnx");
        File.WriteAllBytes(modelPath, new byte[0]);
        
        var options = new InferenceEngineOptions
        {
            ModelProvider = new LocalModelProvider(modelPath)
        };
        
        var engine = new TestEngine(options, NullLogger<TestEngine>.Instance);
        
        // Assert that LoadAsync uses the provided model provider and fails when the model file is invalid.
        await Assert.ThrowsAnyAsync<Exception>(() => engine.LoadAsync());
        engine.Dispose();
    }

    [Fact]
    public async Task LoadAsync_WithoutUrlsOrRegistry_ShouldThrowInvalidOperationException()
    {
        var registryPath = Path.Combine(_testDir, "empty-registry.json");
        File.WriteAllText(registryPath, "{ \"models\": [] }");
        
        var engine = new TestEngineNoUrls(
            new InferenceEngineOptions(), 
            NullLogger<TestEngineNoUrls>.Instance,
            registryPath);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => engine.LoadAsync());
        Assert.Contains("was not found in the ModelRegistry", ex.Message);
        engine.Dispose();
    }

    [Fact]
    public async Task LoadAsync_WithRegistryLookup_ShouldAttemptDownload()
    {
        var registryPath = Path.Combine(_testDir, "test-registry.json");
        var registry = new ModelRegistryData
        {
            Models = new[]
            {
                new ModelRegistryEntry 
                { 
                    Name = "unregistered-model.onnx", 
                    Description = "Test", 
                    Source = "Test",
                    Urls = new[] { "http://localhost:65535/unregistered-model.onnx" } 
                }
            }
        };
        File.WriteAllText(registryPath, System.Text.Json.JsonSerializer.Serialize(registry, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));

        // This engine has no explicit URLs, so it should look up in registry
        var engine = new TestEngineNoUrls(
            new InferenceEngineOptions(), 
            NullLogger<TestEngineNoUrls>.Instance,
            registryPath);

        // The engine should find the URL in the registry and then fail to download.
        // This confirms the registry lookup was successful, as it didn't throw a "not found" error.
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => engine.LoadAsync());
        Assert.Contains("Failed to download model", ex.Message);
        engine.Dispose();
    }

    [Fact]
    public void InferenceEngineOptions_ModelPath_WithLocalModelProvider_ShouldWork()
    {
        var modelPath = Path.Combine(_testDir, "options-test.onnx");
        var options = new InferenceEngineOptions { ModelPath = modelPath };
        
        Assert.NotNull(options.ModelProvider);
        Assert.IsType<LocalModelProvider>(options.ModelProvider);
        Assert.Equal(modelPath, ((LocalModelProvider)options.ModelProvider).ModelPath);
    }

    [Fact]
    public void InferenceEngineOptions_ModelPath_SetToNull_ShouldClearProvider()
    {
        var modelPath = Path.Combine(_testDir, "clear-test.onnx");
        var options = new InferenceEngineOptions { ModelPath = modelPath };
        Assert.NotNull(options.ModelProvider);
        
        options.ModelPath = null;
        Assert.Null(options.ModelProvider);
    }

    [Fact]
    public void InferenceEngineOptions_DefaultValues_ShouldBeCorrect()
    {
        var options = new InferenceEngineOptions();
        
        Assert.True(options.UseGpuAcceleration);
        Assert.Equal(0, options.DeviceId);
        Assert.True(options.WarmupOnLoad);
        Assert.Null(options.ModelProvider);
    }

    // Test helper classes

    private TestEngine CreateTestEngine()
    {
        return new TestEngine(
            new InferenceEngineOptions { ModelPath = "test.onnx" },
            NullLogger<TestEngine>.Instance);
    }

    private class TestEngine : BaseInferenceEngine<string, string>
    {
        public TestEngine(InferenceEngineOptions options, Microsoft.Extensions.Logging.ILogger? logger)
            : base(options, logger) { }

        protected override string ModelName => "test.onnx";
        protected override string[] ModelDownloadUrls => new[] { "http://example.com/test.onnx" };

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
        {
            return (new List<NamedOnnxValue>(), null);
        }

        protected override string PostProcessWithContext(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
        {
            return "test";
        }

        // Expose protected members for testing
        public string GetModelName() => ModelName;
        public string[] GetModelDownloadUrls() => ModelDownloadUrls ?? Array.Empty<string>();
        public string GetModelCacheDirectory() => ModelCacheDirectory;
        public IReadOnlyDictionary<string, NodeMetadata>? GetInputMetadata() => InputMetadata;
        public IReadOnlyDictionary<string, NodeMetadata>? GetOutputMetadata() => OutputMetadata;
        public string? GetPrimaryInputName() => PrimaryInputName;
    }

    private class TestEngineNoUrls : BaseInferenceEngine<string, string>
    {
        private readonly string _registryPath;

        public TestEngineNoUrls(InferenceEngineOptions options, Microsoft.Extensions.Logging.ILogger? logger, string registryPath)
            : base(options, logger)
        {
            _registryPath = registryPath;
        }

        protected override string ModelName => "unregistered-model.onnx";
        // No ModelDownloadUrls override - should fall back to registry
        protected override string[]? ModelDownloadUrls => null;

        protected override ModelRegistry ModelRegistry => new ModelRegistry(_registryPath, _logger);

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
        {
            return (new List<NamedOnnxValue>(), null);
        }

        protected override string PostProcessWithContext(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
        {
            return "test";
        }
    }
}
