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
    private readonly List<TestInfoEngine> _engines = [];

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
    public async Task EngineInfo_ProviderOptions_ReflectsMIGraphXOptions_WhenMIGraphXActive()
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
            Assert.Empty(engine.EngineInfo.ProviderOptions);
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
}
