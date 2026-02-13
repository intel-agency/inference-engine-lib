using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using InferenceEngine.Core.ModelProviders;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace InferenceEngine.Tests;

public class DownloadingModelProviderTests : IDisposable
{
    private readonly string _testDir;

    public DownloadingModelProviderTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"download-tests-{Guid.NewGuid()}");
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            Directory.Delete(_testDir, true);
        }
    }

    [Fact]
    public void Constructor_WithNullModelName_ShouldThrowArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => 
            new DownloadingModelProvider(null!, new[] { "http://example.com/model.onnx" }, _testDir));
    }

    [Fact]
    public void Constructor_WithNullUrls_ShouldThrowArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => 
            new DownloadingModelProvider("model.onnx", null!, _testDir));
    }

    [Fact]
    public void Constructor_WithEmptyUrls_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() => 
            new DownloadingModelProvider("model.onnx", Array.Empty<string>(), _testDir));
    }

    [Fact]
    public async Task GetModelPathAsync_WithCachedModel_ShouldReturnCachedPath()
    {
        var modelName = "cached-model.onnx";
        var cacheDir = Path.Combine(_testDir, "cache1");
        Directory.CreateDirectory(cacheDir);
        var cachedPath = Path.Combine(cacheDir, modelName);
        File.WriteAllText(cachedPath, "cached model content");

        var provider = new DownloadingModelProvider(
            modelName, 
            new[] { "http://example.com/model.onnx" }, 
            cacheDir,
            NullLogger.Instance);

        var result = await provider.GetModelPathAsync();

        Assert.Equal(cachedPath, result);
        Assert.Equal("cached model content", File.ReadAllText(result));
    }

    [Fact]
    public async Task GetModelPathAsync_WithAllUrlsFailing_ShouldThrowInvalidOperationException()
    {
        var modelName = "all-fail.onnx";
        var cacheDir = Path.Combine(_testDir, "cache4");

        var provider = new DownloadingModelProvider(
            modelName, 
            new[] { 
                "http://localhost:65534/fail1.onnx", 
                "http://localhost:65533/fail2.onnx" 
            }, 
            cacheDir,
            NullLogger.Instance);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => provider.GetModelPathAsync());
        Assert.Contains("Failed to download model", ex.Message);
        Assert.Contains("all-fail.onnx", ex.Message);
    }

    [Fact]
    public async Task GetModelPathAsync_CreatesCacheDirectory_IfNotExists()
    {
        var modelName = "newdir-model.onnx";
        var cacheDir = Path.Combine(_testDir, "new", "cache", "dir");

        Assert.False(Directory.Exists(cacheDir));

        // Will fail to download but should create directory first
        var provider = new DownloadingModelProvider(
            modelName, 
            new[] { "http://localhost:65532/test.onnx" }, 
            cacheDir,
            NullLogger.Instance);

        try
        {
            await provider.GetModelPathAsync();
        }
        catch (InvalidOperationException)
        {
            // Expected to fail
        }

        Assert.True(Directory.Exists(cacheDir));
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public async Task GetModelPathAsync_WithRealDownload_ShouldCacheAndReturnPath()
    {
        // Only run when explicitly enabled as an integration test
        var integrationFlag = Environment.GetEnvironmentVariable("INFERENCEENGINE_INTEGRATION_TESTS");
        Skip.IfNot(string.Equals(integrationFlag, "1", StringComparison.Ordinal),
            "Set INFERENCEENGINE_INTEGRATION_TESTS=1 to run this integration test.");
        
        // Skip if network is not available
        Skip.IfNot(NetworkIsAvailable(), "Network is not available");

        var modelName = "yolo11n.onnx";
        var cacheDir = Path.Combine(_testDir, "real-download");
        
        var provider = new DownloadingModelProvider(
            modelName, 
            new[] { "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx" }, 
            cacheDir,
            NullLogger.Instance);

        var result = await provider.GetModelPathAsync();

        Assert.True(File.Exists(result));
        Assert.True(new FileInfo(result).Length > 0);
        Assert.Equal(Path.Combine(cacheDir, modelName), result);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public async Task GetModelPathAsync_WithFirstUrlFailing_ShouldTrySecondUrl()
    {
        // Only run when explicitly enabled as an integration test
        var integrationFlag = Environment.GetEnvironmentVariable("INFERENCEENGINE_INTEGRATION_TESTS");
        Skip.IfNot(string.Equals(integrationFlag, "1", StringComparison.Ordinal),
            "Set INFERENCEENGINE_INTEGRATION_TESTS=1 to run this integration test.");
        
        // Skip if network is not available
        Skip.IfNot(NetworkIsAvailable(), "Network is not available");

        var modelName = "yolo11n.onnx";
        var cacheDir = Path.Combine(_testDir, "failover");
        
        // First URL is invalid, second is valid
        var provider = new DownloadingModelProvider(
            modelName, 
            new[] { 
                "http://localhost:65531/nonexistent.onnx",
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"
            }, 
            cacheDir,
            NullLogger.Instance);

        var result = await provider.GetModelPathAsync();

        Assert.True(File.Exists(result));
        Assert.True(new FileInfo(result).Length > 0);
    }

    [Fact]
    public async Task GetModelPathAsync_WithCancellation_ShouldCancelOperation()
    {
        var modelName = "cancel-model.onnx";
        var cacheDir = Path.Combine(_testDir, "cancel");
        var cts = new CancellationTokenSource();
        
        var provider = new DownloadingModelProvider(
            modelName, 
            new[] { "http://localhost:65530/cancel.onnx" }, 
            cacheDir,
            NullLogger.Instance);

        cts.Cancel();
        
        // Should throw either OperationCanceledException / TaskCanceledException
        // or InvalidOperationException wrapping one of those
        var ex = await Assert.ThrowsAnyAsync<Exception>(() => provider.GetModelPathAsync(cts.Token));

        Assert.True(
            ex is OperationCanceledException or TaskCanceledException ||
            (ex is InvalidOperationException ioe &&
             (ioe.InnerException is OperationCanceledException or TaskCanceledException)),
            "Expected a cancellation-related exception.");
    }

    private static bool NetworkIsAvailable()
    {
        try
        {
            using var client = new System.Net.Http.HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            var response = client.SendAsync(new System.Net.Http.HttpRequestMessage(System.Net.Http.HttpMethod.Head, "https://github.com")).GetAwaiter().GetResult();
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }
}
