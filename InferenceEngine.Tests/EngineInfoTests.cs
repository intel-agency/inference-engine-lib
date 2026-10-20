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

        // The load narrates its phases via Detail before the terminal snapshot.
        Assert.Equal(
            new[] { "Resolving model…", "Selecting execution provider…", "Constructing inference session…" },
            statuses.Where(s => s.State == EngineLoadState.Loading).Select(s => s.Detail).ToArray());
        Assert.Equal(EngineLoadState.Loaded, statuses[^1].State);
        Assert.Null(statuses[^1].Detail);
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

    // W4: platform-gated tests converted from SkippableFact (which permanently skipped on
    // hosts without /dev/kfd, without a GPU, or on the "wrong" OS) into deterministic theory
    // tests driven through the BaseInferenceEngine platform/probe seams via SimulatedPlatformEngine
    // (below). Each theory runs the real provider-selection + session-construction pipeline against
    // a forced platform; only the OS check, KFD check, dlopen probes, and EP append are simulated.
    // Still gated on the ORT native library, which the tests really do use for session construction.
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(3)]
    public async Task EngineInfo_ProviderOptions_ReflectsDeviceId_WhenMIGraphXActive(int deviceId)
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();

        var engine = NewSimulatedEngine(
            new InferenceEngineOptions
            {
                UseGpuAcceleration = true,
                WarmupOnLoad = false,
                DeviceId = deviceId,
                ModelProvider = new LocalModelProvider(modelPath),
            },
            HostPlatform.LinuxX64);

        await engine.LoadAsync();

        // M2: device_id is now surfaced as a real provider option (was a stub returning Empty).
        Assert.Equal(InferenceProvider.MIGraphX, engine.EngineInfo.Provider);
        Assert.NotEmpty(engine.EngineInfo.ProviderOptions);
        Assert.Equal(deviceId.ToString(), engine.EngineInfo.ProviderOptions["device_id"]);
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

    // --- Simulated-platform provider-selection tests (W4: converted from host-gated
    // SkippableFacts that skipped on any CI runner) ---

    [SkippableTheory]
    [InlineData(HostPlatform.LinuxX64)]
    public async Task EngineInfo_RocmBranch_AllPreconditionsPass_MIGraphXWithAllChecksPassed(HostPlatform platform)
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();
        var engine = NewSimulatedEngine(GpuOptions(modelPath), platform,
            kfdPresent: true, allProbesLoadable: true, appendSucceeds: true);

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.MIGraphX, engine.EngineInfo.Provider);
        Assert.False(engine.EngineInfo.IsFallback);
        Assert.Equal(InferenceFallbackReason.None, engine.EngineInfo.FallbackReason);

        foreach (var name in new[] { "kfd.present", "hsa.dlopen", "hip.dlopen", "migraphx.dlopen", "migraphx.append", "migraphx.session" })
        {
            var check = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == name);
            Assert.NotNull(check);
            Assert.Equal(CheckResult.Passed, check!.Result);
        }
    }

    [SkippableTheory]
    [InlineData(HostPlatform.LinuxX64)]
    public async Task EngineInfo_KfdMissing_CpuWithKfdDeviceMissing(HostPlatform platform)
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();
        var engine = NewSimulatedEngine(GpuOptions(modelPath), platform,
            kfdPresent: false, allProbesLoadable: true, appendSucceeds: true);

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.True(engine.EngineInfo.IsFallback);
        Assert.Equal(InferenceFallbackReason.KfdDeviceMissing, engine.EngineInfo.FallbackReason);

        AssertResult("kfd.present", CheckResult.Failed);
        AssertResult("hsa.dlopen", CheckResult.Skipped);   // skipped: precondition failed
        AssertResult("migraphx.session", CheckResult.Skipped);

        void AssertResult(string name, CheckResult expected)
        {
            var check = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == name);
            Assert.NotNull(check);
            Assert.Equal(expected, check!.Result);
        }
    }

    [SkippableTheory]
    [InlineData("libhsa-runtime64.so", InferenceFallbackReason.HsaRuntimeUnloadable, new string[0])]
    [InlineData("libamdhip64.so.7", InferenceFallbackReason.HipRuntimeMissing, new[] { "libhsa-runtime64.so" })]
    [InlineData("libmigraphx_c.so.3", InferenceFallbackReason.MIGraphXRuntimeMissing,
        new[] { "libhsa-runtime64.so", "libamdhip64.so.7" })]
    public async Task EngineInfo_RocmBranch_RuntimeUnloadable_CpuFallbackWithFailingDlopenCheck(
        string failingLibrary, InferenceFallbackReason expectedReason, string[] precedingLibraries)
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();
        var engine = NewSimulatedEngine(GpuOptions(modelPath), HostPlatform.LinuxX64,
            kfdPresent: true,
            probe: name => (name != failingLibrary, (string?)null),
            appendSucceeds: true);

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.True(engine.EngineInfo.IsFallback);
        Assert.Equal(expectedReason, engine.EngineInfo.FallbackReason);

        foreach (var lib in precedingLibraries)
        {
            var checkName = lib switch
            {
                "libhsa-runtime64.so" => "hsa.dlopen",
                "libamdhip64.so.7" => "hip.dlopen",
                _ => throw new ArgumentException(lib),
            };
            var check = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == checkName);
            Assert.NotNull(check);
            Assert.Equal(CheckResult.Passed, check!.Result);
        }

        // The failing probe reports a null error string, so the engine surfaces its built-in
        // detail (install hint naming the library for hip/migraphx, the TryLoad message for hsa).
        var failingCheckName = failingLibrary switch
        {
            "libhsa-runtime64.so" => "hsa.dlopen",
            "libamdhip64.so.7" => "hip.dlopen",
            "libmigraphx_c.so.3" => "migraphx.dlopen",
            _ => throw new ArgumentException(failingLibrary),
        };
        var failingCheck = engine.EngineInfo.Checks.First(c => c.Name == failingCheckName);
        Assert.Equal(CheckResult.Failed, failingCheck.Result);
    }

    [SkippableTheory]
    [InlineData(HostPlatform.WinX64)]
    [InlineData(HostPlatform.WinArm64)]
    public async Task EngineInfo_Windows_DirectMLWithAllChecksPassed(HostPlatform platform)
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();
        var engine = NewSimulatedEngine(GpuOptions(modelPath), platform,
            kfdPresent: true, allProbesLoadable: true, appendSucceeds: true);

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.DirectML, engine.EngineInfo.Provider);
        Assert.False(engine.EngineInfo.IsFallback);

        var dmlCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "dml.append");
        Assert.NotNull(dmlCheck);
        Assert.Equal(CheckResult.Passed, dmlCheck!.Result);
    }

    [SkippableTheory]
    [InlineData(HostPlatform.WinX64)]
    [InlineData(HostPlatform.WinArm64)]
    public async Task EngineInfo_Windows_DirectMLAppendFails_CpuWithProviderAppendFailed(HostPlatform platform)
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();
        var engine = NewSimulatedEngine(GpuOptions(modelPath), platform,
            kfdPresent: true, allProbesLoadable: true, appendSucceeds: false);

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.True(engine.EngineInfo.IsFallback);
        Assert.Equal(InferenceFallbackReason.ProviderAppendFailed, engine.EngineInfo.FallbackReason);

        AssertResult("dml.append", CheckResult.Failed);
        AssertResult("dml.session", CheckResult.Skipped);

        void AssertResult(string name, CheckResult expected)
        {
            var check = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == name);
            Assert.NotNull(check);
            Assert.Equal(expected, check!.Result);
        }
    }

    [SkippableTheory]
    [InlineData(HostPlatform.OSX64)]
    [InlineData(HostPlatform.OSXArm64)]
    public async Task EngineInfo_MacOS_CoreMLWithAllChecksPassed(HostPlatform platform)
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();
        var engine = NewSimulatedEngine(GpuOptions(modelPath), platform,
            kfdPresent: true, allProbesLoadable: true, appendSucceeds: true);

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.CoreML, engine.EngineInfo.Provider);
        Assert.False(engine.EngineInfo.IsFallback);

        var coremlCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "coreml.append");
        Assert.NotNull(coremlCheck);
        Assert.Equal(CheckResult.Passed, coremlCheck!.Result);
    }

    [SkippableTheory]
    [InlineData(HostPlatform.OSX64)]
    [InlineData(HostPlatform.OSXArm64)]
    public async Task EngineInfo_MacOS_CoreMLAppendFails_CpuWithProviderAppendFailed(HostPlatform platform)
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();
        var engine = NewSimulatedEngine(GpuOptions(modelPath), platform,
            kfdPresent: true, allProbesLoadable: true, appendSucceeds: false);

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.True(engine.EngineInfo.IsFallback);
        Assert.Equal(InferenceFallbackReason.ProviderAppendFailed, engine.EngineInfo.FallbackReason);

        var coremlCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "coreml.append");
        Assert.NotNull(coremlCheck);
        Assert.Equal(CheckResult.Failed, coremlCheck!.Result);
    }

    [SkippableTheory]
    [InlineData(HostPlatform.LinuxArm64)]
    [InlineData(HostPlatform.Unknown)]
    public async Task EngineInfo_UnsupportedPlatform_CpuWithUnsupportedPlatformFallback(HostPlatform platform)
    {
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var modelPath = CreateTestModel();
        var engine = NewSimulatedEngine(GpuOptions(modelPath), platform,
            kfdPresent: true, allProbesLoadable: true, appendSucceeds: true);

        await engine.LoadAsync();

        Assert.Equal(InferenceProvider.Cpu, engine.EngineInfo.Provider);
        Assert.True(engine.EngineInfo.IsFallback);
        Assert.Equal(InferenceFallbackReason.UnsupportedPlatform, engine.EngineInfo.FallbackReason);

        var ridCheck = engine.EngineInfo.Checks.FirstOrDefault(c => c.Name == "rid.platform");
        Assert.NotNull(ridCheck);
        Assert.Equal(CheckResult.Failed, ridCheck!.Result);
    }

    private static InferenceEngineOptions GpuOptions(string modelPath) => new()
    {
        UseGpuAcceleration = true,
        WarmupOnLoad = false,
        ModelProvider = new LocalModelProvider(modelPath),
    };

    private SimulatedPlatformEngine NewSimulatedEngine(
        InferenceEngineOptions options, HostPlatform platform,
        bool kfdPresent = true, bool allProbesLoadable = true, bool appendSucceeds = true,
        Func<string, (bool Loaded, string? Error)>? probe = null)
    {
        var engine = new SimulatedPlatformEngine(options, platform, kfdPresent, allProbesLoadable, appendSucceeds, probe);
        _engines.Add(engine);
        return engine;
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

    /// <summary>
    /// Test double that runs the REAL provider-selection and session-construction pipeline against a
    /// SIMULATED host: the platform, KFD device check, dlopen probes, and EP append are forced while
    /// everything else (diagnostic checks, CPU/GPU session options, fallback logic, EngineInfo)
    /// executes for real. This is what lets the provider-branch tests run — not skip — on every OS.
    /// </summary>
    private sealed class SimulatedPlatformEngine : BaseInferenceEngine<string, string>
    {
        private readonly HostPlatform _platform;
        private readonly bool _kfdPresent;
        private readonly bool _allProbesLoadable;
        private readonly bool _appendSucceeds;
        private readonly Func<string, (bool Loaded, string? Error)>? _probe;

        public SimulatedPlatformEngine(
            InferenceEngineOptions options, HostPlatform platform,
            bool kfdPresent, bool allProbesLoadable, bool appendSucceeds,
            Func<string, (bool Loaded, string? Error)>? probe)
            : base(options, NullLogger<SimulatedPlatformEngine>.Instance)
        {
            _platform = platform;
            _kfdPresent = kfdPresent;
            _allProbesLoadable = allProbesLoadable;
            _appendSucceeds = appendSucceeds;
            _probe = probe;
        }

        protected override string ModelName => "simulated-platform.onnx";
        protected override string[]? ModelDownloadUrls => null;

        protected override HostPlatform EffectivePlatform => _platform;
        protected override bool IsKfdPresent() => _kfdPresent;

        protected override bool TryLoadNativeLibrary(string name, out string? error)
        {
            if (_probe is not null)
            {
                (var loaded, error) = _probe(name);
                return loaded;
            }
            error = null;
            return _allProbesLoadable;
        }

        protected override void AppendMIGraphXProvider(SessionOptions options, int deviceId)
        {
            if (!_appendSucceeds) throw new EntryPointNotFoundException("simulated: append failure");
        }

        protected override void AppendDirectMlProvider(SessionOptions options, int deviceId)
        {
            if (!_appendSucceeds) throw new EntryPointNotFoundException("simulated: append failure");
        }

        protected override void AppendCoreMlProvider(SessionOptions options, CoreMLFlags flags)
        {
            if (!_appendSucceeds) throw new EntryPointNotFoundException("simulated: append failure");
        }

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
            => ([], null);

        protected override string PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
            => string.Empty;
    }
}
