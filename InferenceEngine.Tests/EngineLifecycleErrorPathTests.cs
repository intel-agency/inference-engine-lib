using System.Net;
using System.Runtime.InteropServices;
using InferenceEngine.Core;
using InferenceEngine.Core.Diagnostics;
using InferenceEngine.Core.ModelProviders;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace InferenceEngine.Tests;

/// <summary>
/// Error-path and lifecycle coverage for <see cref="BaseInferenceEngine{TInput, TOutput}"/>:
/// warmup success/failure, disposal semantics, provider exceptions surfacing through
/// <see cref="BaseInferenceEngine{TInput, TOutput}.LoadAsync"/> and the status object model,
/// and the hermetic provider-selection fallback branches driven through the existing
/// platform/probe/append/session-construction test seams (no GPU or host platform required).
/// </summary>
public sealed class EngineLifecycleErrorPathTests : IDisposable
{
    private readonly string _testDir;
    private readonly List<IDisposable> _disposables = [];

    public EngineLifecycleErrorPathTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"engine-lifecycle-{Guid.NewGuid()}");
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
        try { Directory.Delete(_testDir, true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }

    // --- Warmup ---

    [SkippableFact]
    public async Task LoadAsync_WithWarmupOnLoad_RunsWarmupAndSucceeds()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreatePlainEngine(warmupOnLoad: true);

        await engine.LoadAsync();

        Assert.Equal(EngineLoadState.Loaded, engine.Status.State);
        Assert.NotNull(engine.InputMetadata);
        Assert.Equal("input", engine.PrimaryInputName);
    }

    [SkippableFact]
    public async Task LoadAsync_WhenWarmupThrows_FailureIsSwallowedAndLoadSucceeds()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = WriteModel();
        var engine = new ThrowingWarmupEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = true,
            ModelProvider = new LocalModelProvider(modelPath),
        });
        _disposables.Add(engine);

        // Warmup failure is non-critical by contract: the engine must still report Loaded.
        await engine.LoadAsync();
        Assert.Equal(EngineLoadState.Loaded, engine.Status.State);
    }

    [Fact]
    public void GetWarmupInput_BeforeLoad_ReturnsEmptyList()
    {
        var engine = CreatePlainEngine(warmupOnLoad: false, withModel: false);
        Assert.Empty(engine.ExposedGetWarmupInput()); // InputMetadata is null pre-load
    }

    // --- Disposal lifecycle ---

    [SkippableFact]
    public async Task Dispose_AfterSuccessfulLoad_ResetsStateAndAllowsDoubleDispose()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreatePlainEngine(warmupOnLoad: false);
        await engine.LoadAsync();
        Assert.Equal(EngineLoadState.Loaded, engine.Status.State);

        engine.Dispose();
        engine.Dispose(); // double dispose must not throw

        Assert.Equal(EngineLoadState.NotLoaded, engine.Status.State);
    }

    [SkippableFact]
    public async Task PredictAsync_AfterDispose_ThrowsObjectDisposedException()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreatePlainEngine(warmupOnLoad: false);
        await engine.LoadAsync();
        engine.Dispose();

        // The disposed load semaphore (not a silent re-load) is the observable contract.
        await Assert.ThrowsAsync<ObjectDisposedException>(() => engine.PredictAsync("x"));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => engine.LoadAsync());
    }

    [Fact]
    public async Task Dispose_DuringLoadIsNotSerialized_PartialInitDoesNotCorruptDispose()
    {
        // Disposing an engine that never loaded is already safe; disposing one whose load FAILED
        // must equally be a no-throw operation (session is null, semaphore released once).
        var engine = new ThrowingProviderEngine();
        _disposables.Add(engine);

        await Assert.ThrowsAsync<InvalidOperationException>(() => engine.LoadAsync());
        engine.Dispose();
        engine.Dispose();
    }

    // --- Model-provider error surfacing ---

    [Fact]
    public async Task LoadAsync_WhenProviderThrows_ExceptionSurfacesAndStatusIsFailed()
    {
        var statuses = new List<InferenceEngineStatus>();
        var engine = new ThrowingProviderEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
            OnStatusChanged = s => statuses.Add(s),
        });
        _disposables.Add(engine);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => engine.LoadAsync());

        Assert.Contains("provider exploded", ex.Message); // original exception, not a wrapper
        Assert.Equal(EngineLoadState.Failed, engine.Status.State);
        Assert.Equal("provider exploded", engine.Status.LoadError);
        Assert.Equal(new[] { EngineLoadState.Loading, EngineLoadState.Failed },
            statuses.Select(s => s.State).ToArray());
    }

    [SkippableFact]
    public async Task LoadAsync_WithFailingThenSucceedingDownloadUrl_LoadsFromSecondUrl()
    {
        // Registry-style multi-URL fallback at the engine level: a DownloadingModelProvider with
        // a stubbed HTTP client whose first URL fails and second URL serves the real model bytes.
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var payload = MinimalOnnxModel.Build();
        var cacheDir = Path.Combine(_testDir, "multi-url-cache");
        var provider = new DownloadingModelProvider(
            "multi-url.onnx",
            ["http://stub.test/primary", "http://stub.test/secondary"],
            cacheDir,
            NullLogger.Instance,
            httpClient: new HttpClient(new StubHandler(req =>
                req.RequestUri!.AbsoluteUri.EndsWith("secondary")
                    ? Ok(new ByteArrayContent(payload))
                    : new HttpResponseMessage(HttpStatusCode.ServiceUnavailable))));

        var engine = new PlainEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
            ModelProvider = provider,
        });
        _disposables.Add(engine);

        await engine.LoadAsync();

        Assert.Equal(EngineLoadState.Loaded, engine.Status.State);
        Assert.Equal(ModelSource.ExplicitPath, engine.EngineInfo.ModelSource);
    }

    // --- Hermetic provider-selection fallback branches (existing T1/T2 seams) ---

    [SkippableFact]
    public async Task Fallback_KfdMissing_CpuWithKfdDeviceMissing()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateSeamEngine(e => { e.Platform = HostPlatform.LinuxX64; e.KfdPresent = false; });

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.Equal(InferenceFallbackReason.KfdDeviceMissing, engine.EngineInfo.FallbackReason);
        AssertResult(engine, "kfd.present", CheckResult.Failed);
        AssertResult(engine, "hsa.dlopen", CheckResult.Skipped); // skipped chain after the failed precondition
        AssertResult(engine, "migraphx.session", CheckResult.Skipped);
    }

    [SkippableFact]
    public async Task Fallback_HsaUnloadable_CpuWithHsaRuntimeUnloadable()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateSeamEngine(e =>
        {
            e.Platform = HostPlatform.LinuxX64;
            e.KfdPresent = true;
            e.FailingLibraries.Add("libhsa-runtime64.so");
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceFallbackReason.HsaRuntimeUnloadable, engine.EngineInfo.FallbackReason);
        AssertResult(engine, "hsa.dlopen", CheckResult.Failed);
        AssertResult(engine, "hip.dlopen", CheckResult.Skipped);
    }

    [SkippableFact]
    public async Task Fallback_HipMissing_CpuWithHipRuntimeMissing()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateSeamEngine(e =>
        {
            e.Platform = HostPlatform.LinuxX64;
            e.KfdPresent = true;
            e.FailingLibraries.Add("libamdhip64.so.7");
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceFallbackReason.HipRuntimeMissing, engine.EngineInfo.FallbackReason);
        AssertResult(engine, "hip.dlopen", CheckResult.Failed);
        Assert.Contains("libamdhip64.so.7", GetCheck(engine, "hip.dlopen")!.Detail);
    }

    [SkippableFact]
    public async Task Fallback_MigraphxRuntimeMissing_CpuWithMIGraphXRuntimeMissing()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateSeamEngine(e =>
        {
            e.Platform = HostPlatform.LinuxX64;
            e.KfdPresent = true;
            e.FailingLibraries.Add("libmigraphx_c.so.3");
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceFallbackReason.MIGraphXRuntimeMissing, engine.EngineInfo.FallbackReason);
        AssertResult(engine, "migraphx.dlopen", CheckResult.Failed);
        Assert.Contains("libmigraphx_c.so.3", GetCheck(engine, "migraphx.dlopen")!.Detail);
    }

    [SkippableFact]
    public async Task Fallback_DirectMlAppendThrows_CpuWithProviderAppendFailed()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateSeamEngine(e =>
        {
            e.Platform = HostPlatform.WinX64;
            e.DirectMlAppendMode = AppendMode.Throw;
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.Equal(InferenceFallbackReason.ProviderAppendFailed, engine.EngineInfo.FallbackReason);
        AssertResult(engine, "dml.append", CheckResult.Failed);
        AssertResult(engine, "dml.session", CheckResult.Skipped);
        AssertResult(engine, "cpu.fallback-session", CheckResult.Passed);
    }

    [SkippableFact]
    public async Task Fallback_CoreMlAppendThrows_CpuWithProviderAppendFailed()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateSeamEngine(e =>
        {
            e.Platform = HostPlatform.OSXArm64;
            e.CoreMlAppendMode = AppendMode.Throw;
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.Equal(InferenceFallbackReason.ProviderAppendFailed, engine.EngineInfo.FallbackReason);
        AssertResult(engine, "coreml.append", CheckResult.Failed);
        AssertResult(engine, "coreml.session", CheckResult.Skipped);
    }

    [SkippableFact]
    public async Task Fallback_UnsupportedPlatform_CpuWithUnsupportedPlatform()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateSeamEngine(e => e.Platform = HostPlatform.LinuxArm64);

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.Equal(InferenceFallbackReason.UnsupportedPlatform, engine.EngineInfo.FallbackReason);
        AssertResult(engine, "rid.platform", CheckResult.Failed);
    }

    [SkippableFact]
    public async Task DirectMl_AppendSucceeds_ProviderIsDirectMlWithSessionCheck()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateSeamEngine(e =>
        {
            e.Platform = HostPlatform.WinX64;
            e.DirectMlAppendMode = AppendMode.NoOp;
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.DirectML, engine.EngineInfo.Provider);
        Assert.False(engine.EngineInfo.IsFallback);
        AssertResult(engine, "dml.append", CheckResult.Passed);
        AssertResult(engine, "dml.session", CheckResult.Passed);
    }

    [SkippableFact]
    public async Task CoreMl_AppendSucceeds_ProviderIsCoreMlWithSessionCheck()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateSeamEngine(e =>
        {
            e.Platform = HostPlatform.OSX64;
            e.CoreMlAppendMode = AppendMode.NoOp;
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.CoreML, engine.EngineInfo.Provider);
        Assert.False(engine.EngineInfo.IsFallback);
        AssertResult(engine, "coreml.append", CheckResult.Passed);
        AssertResult(engine, "coreml.session", CheckResult.Passed);
    }

    [SkippableFact]
    public async Task DirectMl_DefaultAppendRuns_RealAppendOutcomeEitherWay()
    {
        // Exercises the real AppendDirectMlProvider delegate: on non-Windows hosts the native
        // append throws (ProviderAppendFailed fallback); on Windows+DML hosts it succeeds.
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateSeamEngine(e => e.Platform = HostPlatform.WinX64);

        await engine.LoadAsync();

        if (engine.EngineInfo.Provider == InferenceProvider.DirectML)
        {
            AssertResult(engine, "dml.append", CheckResult.Passed);
        }
        else
        {
            Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
            Assert.Equal(InferenceFallbackReason.ProviderAppendFailed, engine.EngineInfo.FallbackReason);
        }
    }

    [SkippableFact]
    public async Task CoreMl_DefaultAppendRuns_RealAppendOutcomeEitherWay()
    {
        // Exercises the real AppendCoreMlProvider delegate: on non-macOS hosts the native
        // append throws (ProviderAppendFailed fallback); on macOS+CoreML hosts it succeeds.
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateSeamEngine(e => e.Platform = HostPlatform.OSXArm64);

        await engine.LoadAsync();

        if (engine.EngineInfo.Provider == InferenceProvider.CoreML)
        {
            AssertResult(engine, "coreml.append", CheckResult.Passed);
        }
        else
        {
            Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
            Assert.Equal(InferenceFallbackReason.ProviderAppendFailed, engine.EngineInfo.FallbackReason);
        }
    }

    [SkippableFact]
    public async Task Fallback_SessionFailureWithRecoverableException_CpuFallback()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateSeamEngine(e =>
        {
            e.Platform = HostPlatform.WinX64;
            e.DirectMlAppendMode = AppendMode.NoOp;
            e.SessionFailuresRemaining = 1; // GPU session construction faults once (recoverable)
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.Equal(InferenceFallbackReason.NativeLibraryLoadFailed, engine.EngineInfo.FallbackReason);
        AssertResult(engine, "dml.session", CheckResult.Failed);
        AssertResult(engine, "session.native-recoverable", CheckResult.Passed);
        AssertResult(engine, "cpu.fallback-session", CheckResult.Passed);
    }

    [SkippableFact]
    public async Task Fallback_SessionFailureWrappedInTypeInitialization_CpuFallback()
    {
        // IsRecoverableNativeFailure must unwrap TypeInitializationException whose inner exception
        // is a recoverable native failure (classic lazy static-init DllNotFoundException).
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var engine = CreateSeamEngine(e =>
        {
            e.Platform = HostPlatform.LinuxX64;
            e.KfdPresent = true;
            e.MigraphxAppendMode = AppendMode.NoOp;
            e.ThrowTypeInitializationOnFirstSession = true;
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.Equal(InferenceFallbackReason.NativeLibraryLoadFailed, engine.EngineInfo.FallbackReason);
        AssertResult(engine, "session.native-recoverable", CheckResult.Passed);
    }

    [SkippableFact]
    public async Task MigraphxCacheDirectory_WhenNotCreatable_ContinuesWithoutCache()
    {
        // The .mxr cache is an optimization: a host that cannot host the cache directory must
        // still reach the provider append (append failures themselves are swallowed by the wrapper).
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var blocker = Path.Combine(_testDir, "cache-blocker.file");
        File.WriteAllText(blocker, "not a directory");

        var engine = new PlainEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(WriteModel()),
            MIGraphXModelCacheDirectory = Path.Combine(blocker, "sub"), // create under a FILE -> IOException
        });
        _disposables.Add(engine);

        // Returns normally: the un-cacheable directory is logged and swallowed inside
        // AppendMIGraphXProvider (the cache is an optimization, never a hard requirement).
        // The EP append itself may fail on hosts without MIGraphX; that failure is also
        // swallowed by the wrapper so the probe is hermetic.
        engine.RunMigraphxAppendWithUncacheableDirectory();
    }

    [Fact]
    public void TryLoadNativeLibrary_WithInvalidName_ReportsFailureWithoutThrowing()
    {
        var engine = CreatePlainEngine(warmupOnLoad: false, withModel: false);
        var loaded = engine.ExposedTryLoadNativeLibrary("bad\0path\0name", out var error);

        // Whether the native loader throws or returns false, the seam must report a failure
        // with a diagnostic, never propagate.
        Assert.False(loaded);
        Assert.False(string.IsNullOrEmpty(error));
    }

    // --- Helpers ---

    private enum AppendMode
    {
        Default,
        NoOp,
        Throw,
    }

    private string WriteModel(MinimalOnnxModel.ElementType elementType = MinimalOnnxModel.ElementType.Float)
    {
        var path = Path.Combine(_testDir, $"model-{Guid.NewGuid():N}.onnx");
        MinimalOnnxModel.Build(path, elementType);
        return path;
    }

    private PlainEngine CreatePlainEngine(bool warmupOnLoad, bool withModel = true)
    {
        var engine = new PlainEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = warmupOnLoad,
            ModelProvider = withModel ? new LocalModelProvider(WriteModel()) : new LocalModelProvider("never-loaded.onnx"),
        });
        _disposables.Add(engine);
        return engine;
    }

    private SeamEngine CreateSeamEngine(Action<SeamEngine> configure)
    {
        var engine = new SeamEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(WriteModel()),
        });
        configure(engine);
        _disposables.Add(engine);
        return engine;
    }

    private static void AssertResult(BaseInferenceEngine<string, string> engine, string checkName, CheckResult expected)
    {
        var check = GetCheck(engine, checkName);
        Assert.NotNull(check);
        Assert.Equal(expected, check!.Result);
    }

    private static ProviderDiagnosticCheck? GetCheck(BaseInferenceEngine<string, string> engine, string checkName)
        => engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == checkName);

    private static bool IsNativeOnnxRuntimeAvailable()
    {
        try
        {
            using var opts = new SessionOptions();
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static HttpResponseMessage Ok(HttpContent content) => new()
    {
        StatusCode = HttpStatusCode.OK,
        Content = content,
    };

    private sealed class StubHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, HttpResponseMessage> _respond;

        public StubHandler(Func<HttpRequestMessage, HttpResponseMessage> respond) => _respond = respond;

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(_respond(request));
    }

    /// <summary>Simplest concrete engine: real CPU session over a hand-built identity model.</summary>
    private sealed class PlainEngine : BaseInferenceEngine<string, string>
    {
        public PlainEngine(InferenceEngineOptions options) : base(options, NullLogger<PlainEngine>.Instance) { }

        protected override string ModelName => "lifecycle-test.onnx";
        protected override string[] ModelDownloadUrls => ["https://example.invalid/lifecycle-test.onnx"];

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => ([NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>([1]))], null);

        protected override string PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
            => string.Empty;

        public List<NamedOnnxValue> ExposedGetWarmupInput() => GetWarmupInput();

        public bool ExposedTryLoadNativeLibrary(string name, out string? error)
            => TryLoadNativeLibrary(name, out error);

        public void RunMigraphxAppendWithUncacheableDirectory()
        {
            try
            {
                using var options = new SessionOptions();
                AppendMIGraphXProvider(options, 0);
            }
            catch (Exception ex) when (ex is EntryPointNotFoundException or DllNotFoundException
                or InvalidOperationException or OnnxRuntimeException)
            {
                // The real EP append failing on hosts without MIGraphX (or whose restored
                // onnxruntime_providers_migraphx library cannot load its own dependencies) is
                // expected; the point of the probe is that the cache-directory failure inside
                // the base was survived.
            }
        }
    }

    /// <summary>Engine whose warmup input construction always throws.</summary>
    private sealed class ThrowingWarmupEngine : BaseInferenceEngine<string, string>
    {
        public ThrowingWarmupEngine(InferenceEngineOptions options) : base(options, NullLogger<ThrowingWarmupEngine>.Instance) { }

        protected override string ModelName => "warmup-throw.onnx";

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => ([], null);

        protected override string PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
            => string.Empty;

        protected override List<NamedOnnxValue> GetWarmupInput()
            => throw new InvalidOperationException("simulated warmup failure");
    }

    /// <summary>Engine whose model provider always throws (hermetic provider-failure path).</summary>
    private sealed class ThrowingProviderEngine : BaseInferenceEngine<string, string>
    {
        public ThrowingProviderEngine(InferenceEngineOptions? options = null)
            : base(Configure(options), NullLogger<ThrowingProviderEngine>.Instance)
        {
        }

        private static InferenceEngineOptions Configure(InferenceEngineOptions? options)
        {
            options ??= new InferenceEngineOptions();
            options.ModelProvider = new ThrowingProvider();
            return options;
        }

        protected override string ModelName => "provider-throw.onnx";

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => ([], null);

        protected override string PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
            => string.Empty;

        private sealed class ThrowingProvider : IModelProvider
        {
            public Task<string> GetModelPathAsync(CancellationToken cancellationToken = default)
                => throw new InvalidOperationException("provider exploded");
        }
    }

    // Type whose static initializer throws a recoverable native failure; the first field access
    // surfaces it wrapped in TypeInitializationException by the runtime.
    private static class StaticInitBoom
    {
        public static readonly int Touch = Throw();

        private static int Throw() => throw new DllNotFoundException("simulated static-init native failure");
    }

    /// <summary>
    /// Test double driving <see cref="BaseInferenceEngine{TInput, TOutput}"/>'s provider-selection
    /// branches deterministically via the platform/probe/append/session-construction seams.
    /// Complements EngineInfoTests' FaultInjectingEngine: it targets the probe-failure and
    /// DirectML/CoreML/unsupported-platform branches that the host-dependent tests cannot reach.
    /// </summary>
    private sealed class SeamEngine : BaseInferenceEngine<string, string>
    {
        public HostPlatform Platform { get; set; } = HostPlatform.LinuxX64;
        public bool KfdPresent { get; set; } = true;
        public HashSet<string> FailingLibraries { get; } = [];
        public bool ForceAllLibrariesLoadable { get; set; } = true;
        public AppendMode MigraphxAppendMode { get; set; } = AppendMode.NoOp;
        public AppendMode DirectMlAppendMode { get; set; } = AppendMode.Default;
        public AppendMode CoreMlAppendMode { get; set; } = AppendMode.Default;
        public int SessionFailuresRemaining { get; set; }
        public bool ThrowTypeInitializationOnFirstSession { get; set; }

        public SeamEngine(InferenceEngineOptions options) : base(options, NullLogger<SeamEngine>.Instance) { }

        protected override string ModelName => "seam-test.onnx";

        protected override HostPlatform EffectivePlatform => Platform;
        protected override bool IsKfdPresent() => KfdPresent;

        protected override bool TryLoadNativeLibrary(string name, out string? error)
        {
            if (FailingLibraries.Contains(name))
            {
                error = $"simulated: {name} not loadable on this host";
                return false;
            }
            if (ForceAllLibrariesLoadable)
            {
                error = null;
                return true;
            }
            return base.TryLoadNativeLibrary(name, out error);
        }

        protected override void AppendMIGraphXProvider(SessionOptions options, int deviceId)
        {
            if (MigraphxAppendMode == AppendMode.Throw)
                throw new EntryPointNotFoundException("simulated: ORT built without the MIGraphX EP");
            if (MigraphxAppendMode == AppendMode.NoOp)
                return; // pretend the append succeeded; session construction is the next step
            base.AppendMIGraphXProvider(options, deviceId);
        }

        protected override void AppendDirectMlProvider(SessionOptions options, int deviceId)
        {
            if (DirectMlAppendMode == AppendMode.Throw)
                throw new DllNotFoundException("simulated: DirectML.dll not present");
            if (DirectMlAppendMode == AppendMode.NoOp)
                return;
            base.AppendDirectMlProvider(options, deviceId);
        }

        protected override void AppendCoreMlProvider(SessionOptions options, CoreMLFlags flags)
        {
            if (CoreMlAppendMode == AppendMode.Throw)
                throw new DllNotFoundException("simulated: CoreML not present");
            if (CoreMlAppendMode == AppendMode.NoOp)
                return;
            base.AppendCoreMlProvider(options, flags);
        }

        protected override InferenceSession CreateInferenceSession(string modelPath, SessionOptions options)
        {
            if (ThrowTypeInitializationOnFirstSession)
            {
                ThrowTypeInitializationOnFirstSession = false;
                _ = StaticInitBoom.Touch; // TypeInitializationException { inner: DllNotFoundException }
            }
            if (SessionFailuresRemaining > 0)
            {
                SessionFailuresRemaining--;
                throw new DllNotFoundException("simulated GPU session native failure");
            }
            return base.CreateInferenceSession(modelPath, options);
        }

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => ([], null);

        protected override string PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
            => string.Empty;
    }
}
