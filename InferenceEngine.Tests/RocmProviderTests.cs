using System.Reflection;
using System.Runtime.InteropServices;
using InferenceEngine.Core;
using InferenceEngine.Core.ModelProviders;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Xunit;

namespace InferenceEngine.Tests;

/// <summary>
/// Level 2 load-time validation for Linux ROCm provider configuration and CPU fallback behavior.
/// Full GPU execution validation is deferred to self-hosted runners with AMD hardware.
/// </summary>
public sealed class RocmProviderTests : IDisposable
{
    private readonly string _testDir;
    private readonly List<TestRocmEngine> _engines = [];

    /// <summary>
    /// Initializes a temporary directory for test artifacts.
    /// </summary>
    public RocmProviderTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"rocm-provider-tests-{Guid.NewGuid()}");
        Directory.CreateDirectory(_testDir);
    }

    /// <summary>
    /// Validates Linux x64 configuration attempts to append ROCm provider when GPU is enabled.
    /// </summary>
    [SkippableFact]
    public void LoadAsync_OnLinuxX64_AttemptsRocmProvider()
    {
        // Platform-specific: this test validates Linux x64 ROCm configuration path only.
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "Test requires Linux x64 platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available in this environment.");
        Skip.IfNot(IsRocmRuntimeUsableOnHost(), "ROCm runtime not usable on this host (no /dev/kfd or libhsa-runtime64.so unreachable).");

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false
        });

        (SessionOptions Options, bool RocmProviderAppended) configuredOptions = default;
        var exception = Record.Exception(() => configuredOptions = engine.GetConfiguredSessionOptions());
        Assert.Null(exception);

        using (configuredOptions.Options)
        {
            Assert.NotNull(configuredOptions.Options);
            Skip.IfNot(
                configuredOptions.RocmProviderAppended,
                "ROCm provider was not appended on this Linux x64 environment; ROCm configuration could not be validated.");
            Assert.True(configuredOptions.RocmProviderAppended);
        }
    }

    /// <summary>
    /// Validates engine load succeeds when ROCm is unavailable by falling back to CPU.
    /// </summary>
    [SkippableFact]
    public async Task LoadAsync_RocmUnavailable_FallsBackToCpu()
    {
        // Platform-specific: this test exercises Linux x64 ROCm-first load with fallback behavior.
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "Test requires Linux x64 platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available in this environment.");

        var modelPath = Path.Combine(_testDir, "minimal-identity.onnx");
        CreateMinimalOnnxModel(modelPath);

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false,
            ModelProvider = new LocalModelProvider(modelPath)
        });

        await engine.LoadAsync();

        Assert.NotNull(engine.InputMetadata);
        Assert.NotNull(engine.OutputMetadata);
        Assert.False(string.IsNullOrWhiteSpace(engine.PrimaryInputName));
    }

    /// <summary>
    /// Validates Linux ARM64 path bypasses ROCm provider append and uses CPU provider.
    /// </summary>
    [SkippableFact]
    public void LoadAsync_OnLinuxArm64_SkipsRocmAttempt()
    {
        // Platform-specific: Linux ARM64 intentionally does not attempt ROCm.
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && RuntimeInformation.ProcessArchitecture == Architecture.Arm64,
            "Test requires Linux ARM64 platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available in this environment.");

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false
        });

        var (options, rocmProviderAppended) = engine.GetConfiguredSessionOptions();
        using (options)
        {
            Assert.False(rocmProviderAppended);
        }
    }

    /// <summary>
    /// Validates Linux x64 ROCm provider configuration executes successfully with a specific device id.
    /// </summary>
    [SkippableFact]
    public void ConfiguredSessionOptions_OnLinuxX64_UsesTypedRocmApi()
    {
        // Platform-specific: validates typed ROCm provider option configuration on Linux x64.
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "Test requires Linux x64 platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available in this environment.");
        Skip.IfNot(IsRocmRuntimeUsableOnHost(), "ROCm runtime not usable on this host (no /dev/kfd or libhsa-runtime64.so unreachable).");

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = true,
            WarmupOnLoad = false,
            DeviceId = 1
        });

        (SessionOptions Options, bool RocmProviderAppended) configuredOptions = default;
        var exception = Record.Exception(() => configuredOptions = engine.GetConfiguredSessionOptions());
        Assert.Null(exception);

        using (configuredOptions.Options)
        {
            Assert.NotNull(configuredOptions.Options);
            Skip.IfNot(
                configuredOptions.RocmProviderAppended,
                "ROCm provider was not appended on this Linux x64 environment; typed ROCm API configuration could not be validated.");
            Assert.True(configuredOptions.RocmProviderAppended);
        }
    }

    /// <summary>
    /// Validates disabling GPU acceleration keeps provider configuration on CPU path.
    /// </summary>
    [SkippableFact]
    public void LoadAsync_GpuDisabled_UsesCpuProvider()
    {
        // Platform-specific: this assertion is Linux-only to stay focused on ROCm-capable OS path.
        Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.Linux), "Test requires Linux platform");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available in this environment.");

        var engine = CreateEngine(new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false
        });

        var (options, rocmProviderAppended) = engine.GetConfiguredSessionOptions();
        using (options)
        {
            Assert.False(rocmProviderAppended);
        }
    }

    /// <summary>
    /// Disposes created engines and removes temporary test artifacts.
    /// </summary>
    public void Dispose()
    {
        foreach (var engine in _engines)
        {
            engine.Dispose();
        }

        if (Directory.Exists(_testDir))
        {
            Directory.Delete(_testDir, true);
        }
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

    /// <summary>
    /// Returns <see langword="true"/> when both the ROCm kernel driver (/dev/kfd)
    /// and the ROCm HSA runtime library (libhsa-runtime64.so) are reachable on the
    /// current host — i.e., the test environment has real ROCm capability.
    /// </summary>
    private static bool IsRocmRuntimeUsableOnHost()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return false;
        if (!File.Exists("/dev/kfd")) return false;
        try
        {
            var handle = NativeLibrary.Load("libhsa-runtime64.so", Assembly.GetExecutingAssembly(), System.Runtime.InteropServices.DllImportSearchPath.UseDllDirectoryForDependencies | System.Runtime.InteropServices.DllImportSearchPath.System32);
            if (handle != IntPtr.Zero) NativeLibrary.Free(handle);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private TestRocmEngine CreateEngine(InferenceEngineOptions options)
    {
        var engine = new TestRocmEngine(options, NullLogger<TestRocmEngine>.Instance);
        _engines.Add(engine);
        return engine;
    }

    private static void CreateMinimalOnnxModel(string path)
    {
        var modelBytes = BuildMinimalIdentityModel();
        File.WriteAllBytes(path, modelBytes);
    }

    private static byte[] BuildMinimalIdentityModel()
    {
        var model = new List<byte>();
        WriteVarintField(model, 1, 9); // ir_version
        WriteStringField(model, 2, "InferenceEngine.Tests");
        WriteLengthDelimitedField(model, 7, BuildGraphProto()); // graph
        WriteLengthDelimitedField(model, 8, BuildOpsetImport()); // opset_import
        return [.. model];
    }

    private static byte[] BuildGraphProto()
    {
        var graph = new List<byte>();
        WriteLengthDelimitedField(graph, 1, BuildIdentityNode()); // node
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
        WriteLengthDelimitedField(typeProto, 1, BuildTensorTypeProto()); // tensor_type
        return [.. typeProto];
    }

    private static byte[] BuildTensorTypeProto()
    {
        var tensorType = new List<byte>();
        WriteVarintField(tensorType, 1, 1); // FLOAT
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
        WriteVarintField(dimension, 1, value); // dim_value
        return [.. dimension];
    }

    private static byte[] BuildOpsetImport()
    {
        var opset = new List<byte>();
        WriteVarintField(opset, 2, 13); // version
        return [.. opset];
    }

    private static void WriteStringField(List<byte> target, int fieldNumber, string value)
    {
        WriteLengthDelimitedField(target, fieldNumber, System.Text.Encoding.UTF8.GetBytes(value));
    }

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
    {
        WriteVarint(target, (ulong)((fieldNumber << 3) | wireType));
    }

    private static void WriteVarint(List<byte> target, ulong value)
    {
        while (value >= 0x80)
        {
            target.Add((byte)((value & 0x7F) | 0x80));
            value >>= 7;
        }

        target.Add((byte)value);
    }

    private sealed class TestRocmEngine : BaseInferenceEngine<string, string>
    {
        public TestRocmEngine(InferenceEngineOptions options, Microsoft.Extensions.Logging.ILogger? logger)
            : base(options, logger)
        {
        }

        protected override string ModelName => "test-rocm.onnx";

        protected override string[] ModelDownloadUrls => ["https://example.invalid/test-rocm.onnx"];

        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
        {
            return ([], null);
        }

        protected override string PostProcessWithContext(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
        {
            return string.Empty;
        }

        public (SessionOptions Options, bool RocmProviderAppended) GetConfiguredSessionOptions()
        {
            var method = typeof(BaseInferenceEngine<string, string>)
                .GetMethod("ConfiguredSessionOptions", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new MissingMethodException(typeof(BaseInferenceEngine<string, string>).FullName, "ConfiguredSessionOptions");

            object?[] parameters = [false];
            var options = method.Invoke(this, parameters) as SessionOptions
                ?? throw new InvalidOperationException("ConfiguredSessionOptions did not return SessionOptions.");

            var rocmProviderAppended = parameters[0] is true;
            return (options, rocmProviderAppended);
        }
    }
}
