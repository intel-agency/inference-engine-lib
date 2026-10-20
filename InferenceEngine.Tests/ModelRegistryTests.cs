using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using InferenceEngine.Core.ModelProviders;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace InferenceEngine.Tests;

public class ModelRegistryTests : IDisposable
{
    private readonly string _testDir;

    public ModelRegistryTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"model-registry-tests-{Guid.NewGuid()}");
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
    public void Constructor_WithValidRegistryFile_ShouldLoadModels()
    {
        var registryPath = CreateTestRegistry(new[]
        {
            new { name = "test1.onnx", description = "Test model 1", source = "Test", urls = new[] { "http://example.com/test1.onnx" } },
            new { name = "test2.onnx", description = "Test model 2", source = "Test", urls = new[] { "http://example.com/test2.onnx", "http://mirror.com/test2.onnx" } }
        });

        var registry = new ModelRegistry(registryPath, NullLogger.Instance);

        Assert.True(registry.IsModelRegistered("test1.onnx"));
        Assert.True(registry.IsModelRegistered("test2.onnx"));
        Assert.False(registry.IsModelRegistered("unknown.onnx"));
    }

    [Fact]
    public void Constructor_WithMissingRegistryFile_ShouldUseEmptyRegistry()
    {
        var missingPath = Path.Combine(_testDir, "nonexistent.json");
        
        var registry = new ModelRegistry(missingPath, NullLogger.Instance);

        Assert.Empty(registry.GetRegisteredModels());
        Assert.False(registry.IsModelRegistered("anything.onnx"));
    }

    [Fact]
    public void Constructor_WithInvalidJson_ShouldUseEmptyRegistry()
    {
        var invalidPath = Path.Combine(_testDir, "invalid.json");
        File.WriteAllText(invalidPath, "not valid json {{{");

        var registry = new ModelRegistry(invalidPath, NullLogger.Instance);

        Assert.Empty(registry.GetRegisteredModels());
    }

    [Fact]
    public void Constructor_WithEmptyModelsArray_ShouldUseEmptyRegistry()
    {
        var registryPath = Path.Combine(_testDir, "empty.json");
        File.WriteAllText(registryPath, "{ \"models\": [] }");

        var registry = new ModelRegistry(registryPath, NullLogger.Instance);

        Assert.Empty(registry.GetRegisteredModels());
    }

    [Fact]
    public void GetModelUrls_WithRegisteredModel_ShouldReturnUrls()
    {
        var registryPath = CreateTestRegistry(new[]
        {
            new { name = "yolo.onnx", description = "YOLO", source = "Test", urls = new[] { "http://github.com/yolo.onnx", "http://huggingface.co/yolo.onnx" } }
        });
        var registry = new ModelRegistry(registryPath, NullLogger.Instance);

        var urls = registry.GetModelUrls("yolo.onnx");

        Assert.NotNull(urls);
        Assert.Equal(2, urls.Length);
        Assert.Contains("http://github.com/yolo.onnx", urls);
        Assert.Contains("http://huggingface.co/yolo.onnx", urls);
    }

    [Fact]
    public void GetModelUrls_WithUnknownModel_ShouldReturnNull()
    {
        var registryPath = CreateTestRegistry(new[]
        {
            new { name = "known.onnx", description = "Known", source = "Test", urls = new[] { "http://example.com/known.onnx" } }
        });
        var registry = new ModelRegistry(registryPath, NullLogger.Instance);

        var urls = registry.GetModelUrls("unknown.onnx");

        Assert.Null(urls);
    }

    [Fact]
    public void GetModelUrls_WithCaseInsensitiveLookup_ShouldWork()
    {
        var registryPath = CreateTestRegistry(new[]
        {
            new { name = "Model.ONNX", description = "Test", source = "Test", urls = new[] { "http://example.com/model.onnx" } }
        });
        var registry = new ModelRegistry(registryPath, NullLogger.Instance);

        var urls = registry.GetModelUrls("model.onnx");

        Assert.NotNull(urls);
        Assert.Single(urls);
    }

    [Fact]
    public void GetRegisteredModels_ShouldReturnAllModelNames()
    {
        var registryPath = CreateTestRegistry(new[]
        {
            new { name = "model1.onnx", description = "Model 1", source = "Test", urls = new[] { "http://example.com/1.onnx" } },
            new { name = "model2.onnx", description = "Model 2", source = "Test", urls = new[] { "http://example.com/2.onnx" } },
            new { name = "model3.onnx", description = "Model 3", source = "Test", urls = new[] { "http://example.com/3.onnx" } }
        });
        var registry = new ModelRegistry(registryPath, NullLogger.Instance);

        var models = registry.GetRegisteredModels();

        Assert.Equal(3, models.Count);
        Assert.Contains("model1.onnx", models);
        Assert.Contains("model2.onnx", models);
        Assert.Contains("model3.onnx", models);
    }

    [Fact]
    public void Constructor_WithEntryMissingName_ShouldSkipEntry()
    {
        var registryPath = Path.Combine(_testDir, "missing-name.json");
        var registry = new ModelRegistryData
        {
            Models = new[]
            {
                new ModelRegistryEntry { Name = "", Description = "Missing name", Urls = new[] { "http://example.com/test.onnx" } },
                new ModelRegistryEntry { Name = "valid.onnx", Description = "Valid", Urls = new[] { "http://example.com/valid.onnx" } },
                new ModelRegistryEntry { Name = null!, Description = "Null name", Urls = new[] { "http://example.com/null.onnx" } }
            }
        };
        File.WriteAllText(registryPath, JsonSerializer.Serialize(registry, new JsonSerializerOptions { WriteIndented = true }));

        var reg = new ModelRegistry(registryPath, NullLogger.Instance);

        Assert.Single(reg.GetRegisteredModels());
        Assert.True(reg.IsModelRegistered("valid.onnx"));
    }

    [Fact]
    public void Constructor_WithEntryMissingUrls_ShouldSkipEntry()
    {
        var registryPath = Path.Combine(_testDir, "missing-urls.json");
        var registry = new ModelRegistryData
        {
            Models = new[]
            {
                new ModelRegistryEntry { Name = "no-urls.onnx", Description = "No URLs", Urls = Array.Empty<string>() },
                new ModelRegistryEntry { Name = "valid.onnx", Description = "Valid", Urls = new[] { "http://example.com/valid.onnx" } },
                new ModelRegistryEntry { Name = "null-urls.onnx", Description = "Null URLs", Urls = null! }
            }
        };
        File.WriteAllText(registryPath, JsonSerializer.Serialize(registry, new JsonSerializerOptions { WriteIndented = true }));

        var reg = new ModelRegistry(registryPath, NullLogger.Instance);

        Assert.Single(reg.GetRegisteredModels());
        Assert.True(reg.IsModelRegistered("valid.onnx"));
    }

    [Fact]
    public void CreateDefaultRegistry_WithNewPath_ShouldCreateFile()
    {
        var registryPath = Path.Combine(_testDir, "default-registry.json");
        
        Assert.False(File.Exists(registryPath));
        
        ModelRegistry.CreateDefaultRegistry(registryPath);
        
        Assert.True(File.Exists(registryPath));
        var content = File.ReadAllText(registryPath);
        Assert.Contains("yolo11n.onnx", content);
        Assert.Contains("huggingface.co", content);
    }

    [Fact]
    public void GetModelEntry_WithSha256InRegistry_ShouldRoundTrip()
    {
        var sha256 = new string('a', 64);
        var registryPath = CreateTestRegistry(new[]
        {
            new { name = "hashed.onnx", description = "Hashed", source = "Test", urls = new[] { "http://example.com/hashed.onnx" }, sha256 }
        });
        var registry = new ModelRegistry(registryPath, NullLogger.Instance);

        var entry = registry.GetModelEntry("hashed.onnx");

        Assert.NotNull(entry);
        Assert.Equal(sha256, entry!.Sha256);
    }

    [Fact]
    public void GetModelEntry_WithoutSha256_ShouldReturnNullSha256()
    {
        var registryPath = CreateTestRegistry(new[]
        {
            new { name = "plain.onnx", description = "Plain", source = "Test", urls = new[] { "http://example.com/plain.onnx" } }
        });
        var registry = new ModelRegistry(registryPath, NullLogger.Instance);

        Assert.Null(registry.GetModelEntry("plain.onnx")!.Sha256);
    }

    [Fact]
    public void GetModelEntry_WithUnknownModel_ShouldReturnNull()
    {
        var registryPath = CreateTestRegistry(new[]
        {
            new { name = "known.onnx", description = "Known", source = "Test", urls = new[] { "http://example.com/known.onnx" } }
        });
        var registry = new ModelRegistry(registryPath, NullLogger.Instance);

        Assert.Null(registry.GetModelEntry("unknown.onnx"));
    }

    [Fact]
    public void Sha256_SerializesAsLowercaseCamelCaseJson()
    {
        var sha256 = new string('b', 64);
        var data = new ModelRegistryData
        {
            Models = new[] { new ModelRegistryEntry { Name = "m.onnx", Urls = new[] { "http://example.com/m.onnx" }, Sha256 = sha256 } }
        };

        var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });

        Assert.Contains("\"sha256\":", json);
        var roundTripped = JsonSerializer.Deserialize<ModelRegistryData>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true // same options the ModelRegistry loader uses
        })!;
        Assert.NotNull(roundTripped.Models);
        Assert.Single(roundTripped.Models);
        Assert.Equal(sha256, roundTripped.Models[0].Sha256);
    }

    [Fact]
    public void CreateDefaultRegistry_WhenFileExists_ShouldNotOverwrite()
    {
        var registryPath = Path.Combine(_testDir, "existing.json");
        File.WriteAllText(registryPath, "{ \"custom\": true }");
        var originalContent = File.ReadAllText(registryPath);

        ModelRegistry.CreateDefaultRegistry(registryPath);

        var newContent = File.ReadAllText(registryPath);
        Assert.Equal(originalContent, newContent);
    }

    // --- W3: remote-fetch failure / fallback corners ---

    [Fact]
    public void Constructor_WithExplicitNullModels_ShouldUseEmptyRegistry()
    {
        var registryPath = Path.Combine(_testDir, "null-models.json");
        File.WriteAllText(registryPath, "{ \"models\": null }");

        var registry = new ModelRegistry(registryPath, NullLogger.Instance);

        Assert.Empty(registry.GetRegisteredModels());
    }

    [SkippableFact]
    public void Constructor_WithUnreadableFile_ShouldUseEmptyRegistry()
    {
        Skip.IfNot(PosixInterop.IsLinux, "Requires POSIX chmod to make the registry file unreadable.");
        Skip.If(PosixInterop.RunningAsRoot, "chmod does not restrict root.");

        var registryPath = Path.Combine(_testDir, "unreadable.json");
        File.WriteAllText(registryPath, "{ \"models\": [] }");
        Assert.True(PosixInterop.TryChmod(registryPath, PosixInterop.None), "chmod must succeed for the test to be meaningful");

        try
        {
            // Non-JSON failure (UnauthorizedAccessException from the read) is caught and degraded
            // to an empty registry rather than thrown from the constructor.
            var registry = new ModelRegistry(registryPath, NullLogger.Instance);
            Assert.Empty(registry.GetRegisteredModels());
        }
        finally
        {
            PosixInterop.TryChmod(registryPath, PosixInterop.OwnerReadWriteExecute);
        }
    }

    [Fact]
    public void CreateDefaultRegistry_WithMissingParentDirectory_ShouldCreateDirectoryAndFile()
    {
        var registryPath = Path.Combine(_testDir, "deep", "nested", "default-registry.json");
        Assert.False(Directory.Exists(Path.GetDirectoryName(registryPath)));

        ModelRegistry.CreateDefaultRegistry(registryPath);

        Assert.True(File.Exists(registryPath));
        Assert.Contains("yolo11n.onnx", File.ReadAllText(registryPath));
    }

    private string CreateTestRegistry<T>(T[] models)
    {
        var path = Path.Combine(_testDir, $"registry-{Guid.NewGuid()}.json");
        var data = new { models };
        File.WriteAllText(path, JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true }));
        return path;
    }
}
