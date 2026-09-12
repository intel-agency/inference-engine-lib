using System.Runtime.InteropServices;
using InferenceEngine.Core;
using InferenceEngine.Core.Diagnostics;
using InferenceEngine.Core.ModelProviders;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Xunit;

namespace InferenceEngine.Tests;

public sealed class EngineInfoTests : IDisposable
{
    private readonly string _testDir;
    private readonly List<BaseInferenceEngine<string, string>> _engines = [];

    public EngineInfoTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"engine-info-tests-{Guid.NewGuid()}");
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        foreach (var engine in _engines)
            engine.Dispose();

        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, true); }
            catch { }
        }
    }

    // --- Unit tests (no GPU/ORT session required) ---

    [Fact]
    public void EngineInfo_BeforeLoad_ReturnsEmpty()
    {
        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
        });

        Assert.Same(InferenceEngineInfo.Empty, engine.EngineInfo);
        Assert.Equal(InferenceProvider.Unknown, engine.EngineInfo.Provider);
        Assert.Empty(engine.EngineInfo.Checks);
    }

    [SkippableFact]
    public async Task EngineInfo_AfterLoad_ReportsModelIdentityAndSource()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
        });

        await engine.LoadAsync();

        Assert.Equal("test-info.onnx", engine.EngineInfo.ModelName);
        Assert.Equal(modelPath, engine.EngineInfo.ModelPath);
        Assert.Equal(ModelSource.ExplicitPath, engine.EngineInfo.ModelSource);
    }

    [Fact]
    public async Task Status_ModelResolutionFailure_IsFailedWithLoadErrorAndLoadingTransition()
    {
        // Hermetic failure: the model is not in the registry and the engine specifies no URLs,
        // so LoadAsync throws before any session/provider work. The status object model must
        // still show the full transition sequence and the error.
        var statuses = new List<InferenceEngineStatus>();
        var options = new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
            OnStatusChanged = s => statuses.Add(s),
        };
        var engine = new UnregisteredModelEngine(options);
        _engines.Add(engine);

        await Assert.ThrowsAsync<InvalidOperationException>(() => engine.LoadAsync());

        Assert.Equal(new[] { EngineLoadState.Loading, EngineLoadState.Failed },
            statuses.Select(s => s.State).ToArray());
        Assert.Equal(EngineLoadState.Failed, engine.Status.State);
        Assert.Contains("not-in-registry.onnx", engine.Status.LoadError);
        Assert.NotNull(engine.Status.Info); // partial info (empty checks: failure predated selection)
    }

    [Fact]
    public async Task Status_ThrowingCallback_IsSwallowedAndDoesNotAffectEngine()
    {
        var options = new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
            OnStatusChanged = _ => throw new FormatException("consumer callback blew up"),
        };
        var engine = new UnregisteredModelEngine(options);
        _engines.Add(engine);

        // The load itself fails (registry miss — InvalidOperationException); the callback's
        // FormatException must NOT replace or mask it.
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => engine.LoadAsync());
        Assert.DoesNotContain("consumer callback", ex.Message);
        Assert.Equal(EngineLoadState.Failed, engine.Status.State);
    }

    [SkippableFact]
    public async Task Status_SuccessfulLoad_TransitionsLoadingThenLoadedWithSameInfo()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();
        var statuses = new List<InferenceEngineStatus>();
        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
            OnStatusChanged = s => statuses.Add(s),
        });

        await engine.LoadAsync();

        Assert.Equal(new[] { EngineLoadState.Loading, EngineLoadState.Loaded },
            statuses.Select(s => s.State).ToArray());
        Assert.Equal(EngineLoadState.Loaded, engine.Status.State);
        Assert.Same(engine.EngineInfo, engine.Status.Info);
        Assert.False(engine.Status.IsGpuActive); // CPU by request
    }


    [SkippableFact]
    public async Task EngineInfo_GpuDisabled_CpuWithUserDisabledGpu()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.True(engine.EngineInfo.IsFallback);
        Assert.Equal(InferenceFallbackReason.UserDisabledGpu, engine.EngineInfo.FallbackReason);
        Assert.False(engine.EngineInfo.GpuAccelerationRequested);

        var gpuCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "gpu.user-enabled");
        Assert.NotNull(gpuCheck);
        Assert.Equal(CheckResult.Failed, gpuCheck.Result);
    }

    [SkippableFact]
    public async Task EngineInfo_Checks_OrderedByOrderProperty()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
        });

        await engine.LoadAsync();

        for (int i = 0; i < engine.EngineInfo.Checks.Count; i++)
        {
            Assert.Equal(i, engine.EngineInfo.Checks[i].Order);
        }
    }

    [SkippableFact]
    public async Task OnSessionInitialized_Callback_FiresOnceWithPopulatedInfo()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();
        InferenceEngineInfo? callbackInfo = null;

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
            OnSessionInitialized = info => callbackInfo = info,
        });

        await engine.LoadAsync();

        Assert.NotNull(callbackInfo);
        Assert.Equal(InferenceProvider.Cpu, callbackInfo.Provider);
        Assert.NotEmpty(callbackInfo.Checks);
        Assert.NotEqual(default, callbackInfo.SessionCreatedAt);
    }

    [SkippableFact]
    public async Task OnSessionInitialized_Callback_Throws_EngineStillUsable()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
            OnSessionInitialized = _ => throw new InvalidOperationException("callback boom"),
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
    }

    [SkippableFact]
    public async Task OnSessionInitialized_Callback_SecondLoadAsync_DoesNotFireAgain()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();
        var callCount = 0;

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
            OnSessionInitialized = _ => callCount++,
        });

        await engine.LoadAsync();
        await engine.LoadAsync();

        Assert.Equal(1, callCount);
    }

    [SkippableFact]
    public async Task EngineInfo_ProviderOptions_ReflectsDeviceId_WhenMIGraphXActive()
    {
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) &&
            RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "Test requires Linux x64 platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");
        Skip.IfNot(IsMIGraphXRuntimeUsableOnHost(), "MIGraphX runtime not usable on this host.");

        var modelPath = CreateTestModel();

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
        });

        await engine.LoadAsync();

        if (engine.EngineInfo.Provider == InferenceProvider.MIGraphX)
        {
            // M2: device_id is now surfaced as a real provider option (was a stub returning Empty).
            Assert.NotEmpty(engine.EngineInfo.ProviderOptions);
            Assert.Equal(engine.EngineInfo.DeviceId.ToString(), engine.EngineInfo.ProviderOptions["device_id"]);
        }
    }

    // T1: the native-recoverable fallback branch — the most complex path — asserts
    // FallbackReason == NativeLibraryLoadFailed plus the session.native-recoverable + cpu.fallback-*
    // checks. Hermetic: the platform/probes/append are forced, and only the GPU session
    // construction is faulted (recoverable), so this runs on any host with the ORT native lib.
    [SkippableFact]
    public async Task Fallback_NativeLibraryLoadFailed_RecoverableCpuFallback()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();

        var engine = new FaultInjectingEngine(
            new InferenceEngineOptions
            {
                UseGpuAcceleration = true,
                WarmupOnLoad = false,
                ModelProvider = new LocalModelProvider(modelPath),
            },
            NullLogger<FaultInjectingEngine>.Instance,
            FaultInjectingEngine.Mode.NativeSessionFailure);
        _engines.Add(engine);

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.True(engine.EngineInfo.IsFallback);
        Assert.Equal(InferenceFallbackReason.NativeLibraryLoadFailed, engine.EngineInfo.FallbackReason);

        AssertResult("migraphx.append", CheckResult.Passed);   // append succeeded (forced)
        AssertResult("migraphx.session", CheckResult.Failed);   // GPU session construction faulted
        AssertResult("session.native-recoverable", CheckResult.Passed);
        AssertResult("cpu.fallback-options", CheckResult.Passed);
        AssertResult("cpu.fallback-session", CheckResult.Passed);

        void AssertResult(string name, CheckResult expected)
        {
            var check = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == name);
            Assert.NotNull(check);
            Assert.Equal(expected, check!.Result);
        }
    }

    // T2: AppendExecutionProvider_MIGraphX throwing after all preconditions pass (e.g. an ORT build
    // without the EP) yields FallbackReason == ProviderAppendFailed with *.append Failed + *.session
    // Skipped, then a successful CPU fallback. Hermetic on any host with the ORT native lib.
    [SkippableFact]
    public async Task Fallback_ProviderAppendFailed_WhenAppendThrows()
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();

        var engine = new FaultInjectingEngine(
            new InferenceEngineOptions
            {
                UseGpuAcceleration = true,
                WarmupOnLoad = false,
                ModelProvider = new LocalModelProvider(modelPath),
            },
            NullLogger<FaultInjectingEngine>.Instance,
            FaultInjectingEngine.Mode.ProviderAppendFailure);
        _engines.Add(engine);

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.True(engine.EngineInfo.IsFallback);
        Assert.Equal(InferenceFallbackReason.ProviderAppendFailed, engine.EngineInfo.FallbackReason);

        AssertResult("migraphx.append", CheckResult.Failed);
        AssertResult("migraphx.session", CheckResult.Skipped);
        AssertResult("cpu.fallback-options", CheckResult.Passed);
        AssertResult("cpu.fallback-session", CheckResult.Passed);

        void AssertResult(string name, CheckResult expected)
        {
            var check = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == name);
            Assert.NotNull(check);
            Assert.Equal(expected, check!.Result);
        }
    }

    // --- Integration tests (SkippableFact for platform-specific) ---

    [SkippableFact]
    public async Task EngineInfo_LinuxX64WithKfd_MIGraphXWithAllChecksPassed()
    {
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) &&
            RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "Test requires Linux x64 platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");
        Skip.IfNot(File.Exists("/dev/kfd"), "/dev/kfd not present on this host.");
        Skip.IfNot(IsMIGraphXRuntimeUsableOnHost(), "MIGraphX runtime not usable on this host.");

        var modelPath = CreateTestModel();

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
        });

        await engine.LoadAsync();

        Skip.IfNot(
            engine.EngineInfo.Provider == InferenceProvider.MIGraphX,
            $"MIGraphX EP not available in this ONNX Runtime build (got {engine.EngineInfo.Provider}, fallback={engine.EngineInfo.FallbackReason})");

        Assert.False(engine.EngineInfo.IsFallback);
        Assert.Equal(InferenceFallbackReason.None, engine.EngineInfo.FallbackReason);

        var kfdCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "kfd.present");
        Assert.NotNull(kfdCheck);
        Assert.Equal(CheckResult.Passed, kfdCheck.Result);

        var hipCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "hip.dlopen");
        Assert.NotNull(hipCheck);
        Assert.Equal(CheckResult.Passed, hipCheck.Result);

        var mgxCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "migraphx.dlopen");
        Assert.NotNull(mgxCheck);
        Assert.Equal(CheckResult.Passed, mgxCheck.Result);
    }

    [SkippableFact]
    public async Task EngineInfo_LinuxX64WithoutKfd_CpuWithKfdDeviceMissing()
    {
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) &&
            RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "Test requires Linux x64 platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");
        Skip.If(File.Exists("/dev/kfd"), "/dev/kfd is present; test requires it to be absent.");

        var modelPath = CreateTestModel();

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.True(engine.EngineInfo.IsFallback);
        Assert.Equal(InferenceFallbackReason.KfdDeviceMissing, engine.EngineInfo.FallbackReason);

        var kfdCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "kfd.present");
        Assert.NotNull(kfdCheck);
        Assert.Equal(CheckResult.Failed, kfdCheck.Result);
    }

    [SkippableFact]
    public async Task EngineInfo_Windows_DirectMLWithAllChecksPassed()
    {
        Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.Windows), "Test requires Windows platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
        });

        await engine.LoadAsync();

        if (engine.EngineInfo.Provider == InferenceProvider.DirectML)
        {
            Assert.False(engine.EngineInfo.IsFallback);
            var dmlCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "dml.append");
            Assert.NotNull(dmlCheck);
            Assert.Equal(CheckResult.Passed, dmlCheck.Result);
        }
    }

    [SkippableFact]
    public async Task EngineInfo_LinuxX64WithKfd_CpuWithHsaRuntimeUnloadable()
    {
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) &&
            RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "Test requires Linux x64 platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");
        Skip.IfNot(File.Exists("/dev/kfd"), "/dev/kfd not present on this host.");
        Skip.If(IsRocmRuntimeUsableOnHost(), "ROCm HSA runtime IS loadable; test requires it to be missing.");

        var modelPath = CreateTestModel();

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.True(engine.EngineInfo.IsFallback);
        Assert.Equal(InferenceFallbackReason.HsaRuntimeUnloadable, engine.EngineInfo.FallbackReason);

        var kfdCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "kfd.present");
        Assert.NotNull(kfdCheck);
        Assert.Equal(CheckResult.Passed, kfdCheck.Result);

        var hsaCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "hsa.dlopen");
        Assert.NotNull(hsaCheck);
        Assert.Equal(CheckResult.Failed, hsaCheck.Result);
    }

    [SkippableFact]
    public async Task EngineInfo_LinuxX64WithKfd_CpuWithHipRuntimeMissing()
    {
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) &&
            RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "Test requires Linux x64 platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");
        Skip.IfNot(File.Exists("/dev/kfd"), "/dev/kfd not present on this host.");
        Skip.IfNot(IsRocmRuntimeUsableOnHost(), "ROCm HSA runtime not usable; test requires it.");
        Skip.If(IsHipRuntimeUsableOnHost(), "HIP runtime IS loadable; test requires it to be missing.");

        var modelPath = CreateTestModel();

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.True(engine.EngineInfo.IsFallback);
        Assert.Equal(InferenceFallbackReason.HipRuntimeMissing, engine.EngineInfo.FallbackReason);

        var hsaCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "hsa.dlopen");
        Assert.NotNull(hsaCheck);
        Assert.Equal(CheckResult.Passed, hsaCheck.Result);

        var hipCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "hip.dlopen");
        Assert.NotNull(hipCheck);
        Assert.Equal(CheckResult.Failed, hipCheck.Result);
        Assert.Contains("libamdhip64.so.7", hipCheck.Detail);
    }

    [SkippableFact]
    public async Task EngineInfo_LinuxX64WithKfd_CpuWithMIGraphXRuntimeMissing()
    {
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) &&
            RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "Test requires Linux x64 platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");
        Skip.IfNot(File.Exists("/dev/kfd"), "/dev/kfd not present on this host.");
        Skip.IfNot(IsRocmRuntimeUsableOnHost(), "ROCm HSA runtime not usable; test requires it.");
        Skip.IfNot(IsHipRuntimeUsableOnHost(), "HIP runtime not usable; test requires it.");
        Skip.If(IsMIGraphXRuntimeUsableOnHost(), "MIGraphX runtime IS loadable; test requires it to be missing.");

        var modelPath = CreateTestModel();

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
        });

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.True(engine.EngineInfo.IsFallback);
        Assert.Equal(InferenceFallbackReason.MIGraphXRuntimeMissing, engine.EngineInfo.FallbackReason);

        var hipCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "hip.dlopen");
        Assert.NotNull(hipCheck);
        Assert.Equal(CheckResult.Passed, hipCheck.Result);

        var mgxCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "migraphx.dlopen");
        Assert.NotNull(mgxCheck);
        Assert.Equal(CheckResult.Failed, mgxCheck.Result);
        Assert.Contains("libmigraphx_c.so.3", mgxCheck.Detail);
    }

    [SkippableFact]
    public async Task EngineInfo_MacOS_CoreMLWithAllChecksPassed()
    {
        Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.OSX), "Test requires macOS platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath),
        });

        await engine.LoadAsync();

        if (engine.EngineInfo.Provider == InferenceProvider.CoreML)
        {
            Assert.False(engine.EngineInfo.IsFallback);
            var coremlCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "coreml.append");
            Assert.NotNull(coremlCheck);
            Assert.Equal(CheckResult.Passed, coremlCheck.Result);
        }
    }

    // --- Helpers ---

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
            || ex is BadImageFormatException
            || ex is InvalidOperationException
            || (ex is System.Reflection.TargetInvocationException { InnerException: Exception inner }
                && (inner is DllNotFoundException or TypeInitializationException or BadImageFormatException or InvalidOperationException)))
        {
            return false;
        }
    }

    private static bool IsRocmRuntimeUsableOnHost()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return false;
        if (!File.Exists("/dev/kfd")) return false;
        try
        {
            var handle = NativeLibrary.Load("libhsa-runtime64.so",
                System.Reflection.Assembly.GetExecutingAssembly(),
                DllImportSearchPath.UseDllDirectoryForDependencies | DllImportSearchPath.System32);
            if (handle != IntPtr.Zero) NativeLibrary.Free(handle);
            return true;
        }
        catch { return false; }
    }

    private static bool IsHipRuntimeUsableOnHost()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return false;
        try
        {
            var handle = NativeLibrary.Load("libamdhip64.so.7",
                System.Reflection.Assembly.GetExecutingAssembly(),
                DllImportSearchPath.UseDllDirectoryForDependencies | DllImportSearchPath.System32);
            if (handle != IntPtr.Zero) NativeLibrary.Free(handle);
            return true;
        }
        catch { return false; }
    }

    private static bool IsMIGraphXRuntimeUsableOnHost()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return false;
        if (!IsRocmRuntimeUsableOnHost()) return false;
        if (!IsHipRuntimeUsableOnHost()) return false;
        try
        {
            var handle = NativeLibrary.Load("libmigraphx_c.so.3",
                System.Reflection.Assembly.GetExecutingAssembly(),
                DllImportSearchPath.UseDllDirectoryForDependencies | DllImportSearchPath.System32);
            if (handle != IntPtr.Zero) NativeLibrary.Free(handle);
            return true;
        }
        catch { return false; }
    }

    private string CreateTestModel()
    {
        var modelPath = Path.Combine(_testDir, $"test-{Guid.NewGuid()}.onnx");
        File.WriteAllBytes(modelPath, BuildMinimalIdentityModel());
        return modelPath;
    }

    private TestInfoEngine CreateEngine(InferenceEngineOptions options)
    {
        var engine = new TestInfoEngine(options, NullLogger<TestInfoEngine>.Instance);
        _engines.Add(engine);
        return engine;
    }

    private static byte[] BuildMinimalIdentityModel()
    {
        var model = new List<byte>();
        WriteVarintField(model, 1, 9);
        WriteStringField(model, 2, "InferenceEngine.Tests");
        WriteLengthDelimitedField(model, 7, BuildGraphProto());
        WriteLengthDelimitedField(model, 8, BuildOpsetImport());
        return [.. model];
    }

    private static byte[] BuildGraphProto()
    {
        var graph = new List<byte>();
        WriteLengthDelimitedField(graph, 1, BuildIdentityNode());
        WriteStringField(graph, 2, "minimal-identity");
        WriteLengthDelimitedField(graph, 11, BuildValueInfo("input"));
        WriteLengthDelimitedField(graph, 12, BuildValueInfo("output"));
        return [.. graph];
    }

    private static byte[] BuildIdentityNode()
    {
        var node = new List<byte>();
        WriteStringField(node, 1, "input");
        WriteStringField(node, 2, "output");
        WriteStringField(node, 4, "Identity");
        return [.. node];
    }

    private static byte[] BuildValueInfo(string name)
    {
        var valueInfo = new List<byte>();
        WriteStringField(valueInfo, 1, name);
        WriteLengthDelimitedField(valueInfo, 2, BuildTypeProto());
        return [.. valueInfo];
    }

    private static byte[] BuildTypeProto()
    {
        var typeProto = new List<byte>();
        WriteLengthDelimitedField(typeProto, 1, BuildTensorTypeProto());
        return [.. typeProto];
    }

    private static byte[] BuildTensorTypeProto()
    {
        var tensorType = new List<byte>();
        WriteVarintField(tensorType, 1, 1);
        WriteLengthDelimitedField(tensorType, 2, BuildShapeProto());
        return [.. tensorType];
    }

    private static byte[] BuildShapeProto()
    {
        var shape = new List<byte>();
        WriteLengthDelimitedField(shape, 1, BuildDimensionProto(1));
        return [.. shape];
    }

    private static byte[] BuildDimensionProto(ulong value)
    {
        var dimension = new List<byte>();
        WriteVarintField(dimension, 1, value);
        return [.. dimension];
    }

    private static byte[] BuildOpsetImport()
    {
        var opset = new List<byte>();
        WriteVarintField(opset, 2, 13);
        return [.. opset];
    }

    private static void WriteStringField(List<byte> target, int fieldNumber, string value)
        => WriteLengthDelimitedField(target, fieldNumber, System.Text.Encoding.UTF8.GetBytes(value));

    private static void WriteLengthDelimitedField(List<byte> target, int fieldNumber, byte[] payload)
    {
        WriteTag(target, fieldNumber, wireType: 2);
        WriteVarint(target, (ulong)payload.Length);
        target.AddRange(payload);
    }

    private static void WriteVarintField(List<byte> target, int fieldNumber, ulong value)
    {
        WriteTag(target, fieldNumber, wireType: 0);
        WriteVarint(target, value);
    }

    private static void WriteTag(List<byte> target, int fieldNumber, int wireType)
        => WriteVarint(target, (ulong)((fieldNumber << 3) | wireType));

    private static void WriteVarint(List<byte> target, ulong value)
    {
        while (value >= 0x80)
        {
            target.Add((byte)((value & 0x7F) | 0x80));
            value >>= 7;
        }
        target.Add((byte)value);
    }

    private sealed class TestInfoEngine : BaseInferenceEngine<string, string>
    {
        public TestInfoEngine(InferenceEngineOptions options, ILogger? logger)
            : base(options, logger) { }

        protected override string ModelName => "test-info.onnx";
        protected override string[] ModelDownloadUrls => ["https://example.invalid/test-info.onnx"];

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => ([], null);

        protected override string PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
            => string.Empty;
    }

    /// <summary>Engine whose model is neither registered nor downloadable — drives the hermetic
    /// model-resolution failure path (no network, no native session).</summary>
    private sealed class UnregisteredModelEngine : BaseInferenceEngine<string, string>
    {
        public UnregisteredModelEngine(InferenceEngineOptions options)
            : base(options, NullLogger<UnregisteredModelEngine>.Instance) { }

        protected override string ModelName => "not-in-registry.onnx";

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => ([], null);

        protected override string PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
            => string.Empty;
    }

    /// <summary>
    /// Test double that overrides the platform/probe/append/session-construction seams to drive the
    /// complex fallback branches deterministically, on any host (no GPU required). It forces the
    /// Linux-x64/MIGraphX branch, reports all native probes as loadable, and either makes the GPU
    /// session construction throw a recoverable failure (T1) or the EP append throw (T2).
    /// </summary>
    private sealed class FaultInjectingEngine : BaseInferenceEngine<string, string>
    {
        public enum Mode { NativeSessionFailure, ProviderAppendFailure }

        private readonly Mode _mode;
        private int _sessionCalls;

        public FaultInjectingEngine(InferenceEngineOptions options, ILogger? logger, Mode mode)
            : base(options, logger)
        {
            _mode = mode;
        }

        protected override string ModelName => "fault-inject.onnx";
        protected override string[]? ModelDownloadUrls => null;

        // Force the MIGraphX branch regardless of the real host platform.
        protected override HostPlatform EffectivePlatform => HostPlatform.LinuxX64;
        protected override bool IsKfdPresent() => true;
        protected override bool TryLoadNativeLibrary(string name, out string? error)
        {
            error = null;
            return true;
        }

        protected override void AppendMIGraphXProvider(SessionOptions options, int deviceId)
        {
            if (_mode == Mode.ProviderAppendFailure)
                throw new EntryPointNotFoundException("simulated: ORT built without the MIGraphX EP");
            // NativeSessionFailure mode: pretend the append succeeded (do not call the real append,
            // which would throw on non-ROCm hosts) so the branch returns MIGraphX and the GPU
            // session construction is the step that faults.
        }

        protected override InferenceSession CreateInferenceSession(string modelPath, SessionOptions options)
        {
            // T1: the GPU session construction (first call) throws a recoverable native failure;
            // the CPU fallback session (second call) is built for real.
            if (_mode == Mode.NativeSessionFailure && _sessionCalls == 0)
            {
                _sessionCalls++;
                throw new DllNotFoundException("simulated native load failure");
            }
            _sessionCalls++;
            return base.CreateInferenceSession(modelPath, options);
        }

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => ([], null);

        protected override string PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
            => string.Empty;
    }
}
