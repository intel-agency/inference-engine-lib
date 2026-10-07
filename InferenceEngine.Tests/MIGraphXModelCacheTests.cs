using System.Reflection;
using System.Runtime.InteropServices;
using InferenceEngine.Core;
using InferenceEngine.Core.Diagnostics;
using InferenceEngine.Core.Worker;
using Microsoft.ML.OnnxRuntime;
using Xunit;

namespace InferenceEngine.Tests;

/// <summary>
/// Validation for the MIGraphX compiled-model (.mxr) cache wiring: cache-directory
/// resolution rules, provider-option surfacing, the worker-protocol round trip, and —
/// on ROCm-capable hosts — the real provider append. Without the cache option the ORT
/// MIGraphX EP recompiles the full model on every session construction; with it, later
/// startups load the persisted .mxr instead.
/// </summary>
public sealed class MIGraphXModelCacheTests : IDisposable
{
    private readonly List<TestCacheEngine> _engines = [];

    [Fact]
    public void Resolve_DefaultOptions_ReturnsUserProfileDefault()
    {
        // No filesystem side effects are asserted here: other tests in the suite (the
        // out-of-process workers) legitimately create the default directory via the new
        // cache path, so its presence is environment-dependent. Creation-at-append-time is
        // covered by AppendMIGraphXProvider_OnRocmHost_AppendsAndCreatesCacheDirectory.
        var engine = CreateEngine(new InferenceEngineOptions());

        var resolved = engine.ResolveCacheDirectory();

        Assert.NotNull(resolved);
        Assert.Equal(
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "InferenceEngine", "MIGraphXModelCache"),
            resolved);
    }

    [Fact]
    public void Resolve_ExplicitDirectory_WinsVerbatim()
    {
        var engine = CreateEngine(new InferenceEngineOptions
        {
            MIGraphXModelCacheDirectory = Path.Combine(Path.GetTempPath(), "mgx-explicit-cache"),
        });

        Assert.Equal(Path.Combine(Path.GetTempPath(), "mgx-explicit-cache"), engine.ResolveCacheDirectory());
    }

    [Fact]
    public void Resolve_EmptyString_DisablesCaching()
    {
        var engine = CreateEngine(new InferenceEngineOptions { MIGraphXModelCacheDirectory = "" });

        Assert.Null(engine.ResolveCacheDirectory());
    }

    [Fact]
    public void ExtractProviderOptions_MIGraphX_SurfacesCacheDirectory()
    {
        var cacheDir = Path.Combine(Path.GetTempPath(), $"mgx-info-cache-{Guid.NewGuid()}");
        var engine = CreateEngine(new InferenceEngineOptions { MIGraphXModelCacheDirectory = cacheDir });

        var extracted = engine.ExtractProviderOptionsForTest(InferenceProvider.MIGraphX, deviceId: 3);

        Assert.Equal("3", extracted["device_id"]);
        Assert.Equal(cacheDir, extracted["migraphx_model_cache_dir"]);
    }

    [Fact]
    public void ExtractProviderOptions_MIGraphXCacheDisabled_OmitsCacheEntry()
    {
        var engine = CreateEngine(new InferenceEngineOptions { MIGraphXModelCacheDirectory = "" });

        var extracted = engine.ExtractProviderOptionsForTest(InferenceProvider.MIGraphX, deviceId: 0);

        Assert.Equal("0", extracted["device_id"]);
        Assert.False(extracted.ContainsKey("migraphx_model_cache_dir"));
    }

    [Fact]
    public void ExtractProviderOptions_Cpu_StaysEmpty()
    {
        var engine = CreateEngine(new InferenceEngineOptions());

        Assert.Empty(engine.ExtractProviderOptionsForTest(InferenceProvider.Cpu, deviceId: 0));
    }

    [Fact]
    public async Task WorkerEngineOptions_RoundTripsCacheDirectory()
    {
        var envelope = new WorkerEnvelope
        {
            Kind = "load",
            Options = new WorkerEngineOptions { MIGraphXModelCacheDirectory = "/tmp/worker-mgx-cache" },
        };

        using var stream = new MemoryStream();
        WorkerFrame.Write(stream, envelope);
        stream.Position = 0;
        var decoded = await WorkerFrame.ReadAsync(stream, CancellationToken.None);

        Assert.NotNull(decoded?.Options);
        Assert.Equal("/tmp/worker-mgx-cache", decoded.Options.MIGraphXModelCacheDirectory);
    }

    /// <summary>
    /// On ROCm-capable hosts the real append must succeed with the cache option and leave the
    /// cache directory created. Skipped elsewhere (CI has no AMD GPU); the pure rules above run
    /// everywhere.
    /// </summary>
    [SkippableFact]
    public void AppendMIGraphXProvider_OnRocmHost_AppendsAndCreatesCacheDirectory()
    {
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "Test requires Linux x64 platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available in this environment.");
        Skip.IfNot(File.Exists("/dev/kfd"), "ROCm not usable on this host (no /dev/kfd).");

        var cacheDir = Path.Combine(Path.GetTempPath(), $"mgx-append-cache-{Guid.NewGuid()}");
        var engine = CreateEngine(new InferenceEngineOptions { MIGraphXModelCacheDirectory = cacheDir });

        using var options = new SessionOptions();
        var exception = Record.Exception(() => engine.AppendMIGraphXProviderForTest(options, deviceId: 0));

        Assert.Null(exception);
        Assert.True(Directory.Exists(cacheDir), "append must create the cache directory before handing it to the EP");
    }

    public void Dispose()
    {
        foreach (var engine in _engines)
        {
            engine.Dispose();
        }
    }

    private TestCacheEngine CreateEngine(InferenceEngineOptions options)
    {
        var engine = new TestCacheEngine(options);
        _engines.Add(engine);
        return engine;
    }

    /// <summary>
    /// Returns <see langword="true"/> when the ONNX Runtime native library can be loaded;
    /// <see langword="false"/> when the library is absent or has unresolvable dependencies.
    /// </summary>
    private static bool IsNativeOnnxRuntimeAvailable()
    {
        try
        {
            using var opts = new SessionOptions();
            return true;
        }
        catch (Exception ex) when (
            ex is DllNotFoundException
            || ex is TypeInitializationException
            || (ex is TargetInvocationException { InnerException: Exception inner }
                && (inner is DllNotFoundException or TypeInitializationException)))
        {
            return false;
        }
    }

    /// <summary>Exposes the protected cache seams for testing without reflection.</summary>
    private sealed class TestCacheEngine : BaseInferenceEngine<string, string>
    {
        public TestCacheEngine(InferenceEngineOptions options)
            : base(options)
        {
        }

        protected override string ModelName => "test-mgx-cache.onnx";

        protected override string[] ModelDownloadUrls => ["https://example.invalid/test-mgx-cache.onnx"];

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => ([], null);

        protected override string PostProcessWithContext(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
            => string.Empty;

        public string? ResolveCacheDirectory() => ResolveMIGraphXModelCacheDirectory();

        public void AppendMIGraphXProviderForTest(SessionOptions options, int deviceId)
            => AppendMIGraphXProvider(options, deviceId);

        public IReadOnlyDictionary<string, string> ExtractProviderOptionsForTest(InferenceProvider provider, int deviceId)
        {
            var method = typeof(BaseInferenceEngine<string, string>)
                .GetMethod("ExtractProviderOptions", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new MissingMethodException(typeof(BaseInferenceEngine<string, string>).FullName, "ExtractProviderOptions");

            return method.Invoke(this, [provider, deviceId]) as IReadOnlyDictionary<string, string>
                ?? throw new InvalidOperationException("ExtractProviderOptions did not return a dictionary.");
        }
    }
}
