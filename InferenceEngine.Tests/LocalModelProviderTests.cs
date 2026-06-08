using System;
using System.IO;
using System.Threading.Tasks;
using InferenceEngine.Core.ModelProviders;
using Xunit;

namespace InferenceEngine.Tests;

public class LocalModelProviderTests : IDisposable
{
    private readonly string _testDir;

    public LocalModelProviderTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"local-provider-tests-{Guid.NewGuid()}");
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
    public void Constructor_WithNullPath_ShouldThrowArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new LocalModelProvider(null!));
    }

    [Fact]
    public void Constructor_WithEmptyPath_ShouldNotThrow()
    {
        // Empty string is allowed at construction time, validation happens at GetModelPathAsync
        var provider = new LocalModelProvider(string.Empty);
        Assert.NotNull(provider);
    }

    [Fact]
    public async Task GetModelPathAsync_WithExistingFile_ShouldReturnFullPath()
    {
        var modelPath = Path.Combine(_testDir, "test-model.onnx");
        File.WriteAllText(modelPath, "fake model content");
        var provider = new LocalModelProvider(modelPath);

        var result = await provider.GetModelPathAsync();

        Assert.Equal(Path.GetFullPath(modelPath), result);
        Assert.True(File.Exists(result));
    }

    [Fact]
    public async Task GetModelPathAsync_WithRelativePath_ShouldResolveToFullPath()
    {
        var originalDir = Environment.CurrentDirectory;
        try
        {
            Environment.CurrentDirectory = _testDir;
            var modelPath = "relative-model.onnx";
            File.WriteAllText(Path.Combine(_testDir, modelPath), "fake model content");
            var provider = new LocalModelProvider(modelPath);

            var result = await provider.GetModelPathAsync();

            Assert.True(Path.IsPathFullyQualified(result));
            Assert.True(File.Exists(result));
        }
        finally
        {
            Environment.CurrentDirectory = originalDir;
        }
    }

    [Fact]
    public async Task GetModelPathAsync_WithNonExistingFile_ShouldThrowFileNotFoundException()
    {
        var modelPath = Path.Combine(_testDir, "non-existent.onnx");
        var provider = new LocalModelProvider(modelPath);

        var ex = await Assert.ThrowsAsync<FileNotFoundException>(() => provider.GetModelPathAsync());
        Assert.Contains("non-existent.onnx", ex.Message);
    }

    [Fact]
    public async Task GetModelPathAsync_ShouldReturnSamePathOnMultipleCalls()
    {
        var modelPath = Path.Combine(_testDir, "consistent.onnx");
        File.WriteAllText(modelPath, "fake model content");
        var provider = new LocalModelProvider(modelPath);

        var result1 = await provider.GetModelPathAsync();
        var result2 = await provider.GetModelPathAsync();

        Assert.Equal(result1, result2);
    }

    [Fact]
    public void ModelPath_Property_ShouldReturnOriginalPath()
    {
        var modelPath = Path.Combine(_testDir, "property-test.onnx");
        var provider = new LocalModelProvider(modelPath);

        Assert.Equal(modelPath, provider.ModelPath);
    }

    [Fact]
    public async Task GetModelPathAsync_WithNestedDirectory_ShouldWork()
    {
        var nestedDir = Path.Combine(_testDir, "nested", "deep");
        Directory.CreateDirectory(nestedDir);
        var modelPath = Path.Combine(nestedDir, "deep-model.onnx");
        File.WriteAllText(modelPath, "fake model content");
        var provider = new LocalModelProvider(modelPath);

        var result = await provider.GetModelPathAsync();

        Assert.True(File.Exists(result));
    }
}
