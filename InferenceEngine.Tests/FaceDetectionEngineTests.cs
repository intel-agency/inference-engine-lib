using System.Net;
using System.Reflection;
using System.Runtime.InteropServices;
using InferenceEngine.Core;
using InferenceEngine.Examples;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Xunit;

namespace InferenceEngine.Tests;

public class FaceDetectionEngineTests
{
    private static readonly HttpClient Http = new();

    private static bool NetworkIsAvailable()
    {
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            var response = client.SendAsync(new HttpRequestMessage(HttpMethod.Head, "https://github.com")).Result;
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    private static async Task<byte[]> DownloadImageAsync(string url)
    {
        using var resp = await Http.GetAsync(url);
        if (resp.StatusCode != HttpStatusCode.OK)
        {
            throw new InvalidOperationException($"Failed to download '{url}'. Status: {(int)resp.StatusCode} {resp.ReasonPhrase}");
        }
        return await resp.Content.ReadAsByteArrayAsync();
    }

    /// <summary>
    /// Creates a FaceDetectionEngine with default options.
    /// The engine automatically handles model acquisition via CreateDefaultModelProvider().
    /// </summary>
    private static FaceDetectionEngine CreateEngine(bool useGpu = false)
    {
        return new FaceDetectionEngine(
            new InferenceEngineOptions
            {
                UseGpuAcceleration = useGpu,
                WarmupOnLoad = false
            },
            NullLogger<FaceDetectionEngine>.Instance);
    }

    [SkippableFact]
    public async Task LoadAsync_WithDefaultProvider_ShouldPopulateMetadata()
    {
        Skip.IfNot(NetworkIsAvailable(), "Network is not available");
        Skip.IfNot(IsBaseNativeOnnxRuntimeLoadable(), "Base ONNX Runtime native library is not loadable — verify libonnxruntime.so is present and built for linux-x64 (try --runtime linux-x64)");
        var engine = CreateEngine();

        await engine.LoadAsync();

        Assert.NotNull(engine.InputMetadata);
        Assert.NotNull(engine.OutputMetadata);
        Assert.False(string.IsNullOrWhiteSpace(engine.PrimaryInputName));
    }

    [SkippableFact]
    public async Task PredictAsync_WithWebImage_ShouldReturnSomePredictions()
    {
        Skip.IfNot(NetworkIsAvailable(), "Network is not available");
        Skip.IfNot(IsBaseNativeOnnxRuntimeLoadable(), "Base ONNX Runtime native library is not loadable — verify libonnxruntime.so is present and built for linux-x64 (try --runtime linux-x64)");
        var imageBytes = await DownloadImageAsync("https://ultralytics.com/images/bus.jpg");
        var engine = CreateEngine();

        var results = await engine.PredictAsync(imageBytes);

        Assert.NotNull(results);
        Assert.NotEmpty(results);
    }

    [SkippableFact]
    public async Task PredictAsync_WithBusImage_ShouldDetectAtLeastOnePersonOrBusOrCar()
    {
        Skip.IfNot(NetworkIsAvailable(), "Network is not available");
        Skip.IfNot(IsBaseNativeOnnxRuntimeLoadable(), "Base ONNX Runtime native library is not loadable — verify libonnxruntime.so is present and built for linux-x64 (try --runtime linux-x64)");
        var imageBytes = await DownloadImageAsync("https://ultralytics.com/images/bus.jpg");
        var engine = CreateEngine();

        var results = await engine.PredictAsync(imageBytes);

        Assert.Contains(results, r => r.Label is "person" or "bus" or "car");
    }

    // --- ROCm GPU Inference Tests ---

    /// <summary>
    /// Returns true when <c>libonnxruntime_providers_migraphx.so</c> is present in the output
    /// directory, indicating the <c>InferenceEngine.ROCm.Runtime.linux-x64</c> NuGet package
    /// has been restored and its native assets were copied to the build output.
    /// </summary>
    private static bool IsMIGraphXLibPresent()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "libonnxruntime_providers_migraphx.so"),
            Path.Combine(AppContext.BaseDirectory, "runtimes", "linux-x64", "native", "libonnxruntime_providers_migraphx.so"),
        };
        return candidates.Any(File.Exists);
    }

    /// <summary>
    /// Returns <see langword="true"/> when the base ONNX Runtime native library
    /// (<c>libonnxruntime.so</c> on Linux) can be loaded. Catches the
    /// <see cref="DllNotFoundException"/> / <see cref="TypeInitializationException"/>
    /// shapes that arise when the native runtime is missing or built for the wrong platform,
    /// so callers can produce a clear, distinguishable failure message rather than
    /// letting a deep <c>OnnxRuntimeException</c> bubble out of <c>LoadAsync</c>.
    /// </summary>
    private static bool IsBaseNativeOnnxRuntimeLoadable()
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
    /// Returns true when an AMD GPU is accessible via the ROCm kernel driver
    /// (<c>/dev/kfd</c> is present) AND the ROCm HSA runtime library
    /// (<c>libhsa-runtime64.so</c>) is resolvable via dlopen. Mirrors the production
    /// check in <c>BaseInferenceEngine.IsRocmRuntimeUsable()</c>.
    /// </summary>
    private static bool IsRocmRuntimeUsableOnHost()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return false;
        if (!File.Exists("/dev/kfd")) return false;
        try
        {
            var handle = NativeLibrary.Load(
                "libhsa-runtime64.so",
                Assembly.GetExecutingAssembly(),
                DllImportSearchPath.UseDllDirectoryForDependencies | DllImportSearchPath.System32);
            if (handle != IntPtr.Zero) NativeLibrary.Free(handle);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool IsMIGraphXRuntimeUsableOnHost()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return false;
        try
        {
            var hipHandle = NativeLibrary.Load("libamdhip64.so.7",
                Assembly.GetExecutingAssembly(),
                DllImportSearchPath.UseDllDirectoryForDependencies | DllImportSearchPath.System32);
            if (hipHandle != IntPtr.Zero) NativeLibrary.Free(hipHandle);
        }
        catch { return false; }

        try
        {
            var mgxHandle = NativeLibrary.Load("libmigraphx_c.so.3",
                Assembly.GetExecutingAssembly(),
                DllImportSearchPath.UseDllDirectoryForDependencies | DllImportSearchPath.System32);
            if (mgxHandle != IntPtr.Zero) NativeLibrary.Free(mgxHandle);
            return true;
        }
        catch { return false; }
    }

    /// <summary>
    /// Validates end-to-end inference runs successfully on Linux x64 when the ROCm native
    /// library is present. The engine may use the ROCm EP or fall back to CPU transparently;
    /// either path must produce valid detections.
    /// </summary>
    [SkippableFact]
    public async Task PredictAsync_OnLinuxX64WithRocmLib_ShouldSucceedEndToEnd()
    {
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "Test requires Linux x64 platform");
        Skip.IfNot(IsMIGraphXLibPresent(), "libonnxruntime_providers_migraphx.so not found — InferenceEngine.ROCm.Runtime.linux-x64 package assets must be present");
        Skip.IfNot(IsBaseNativeOnnxRuntimeLoadable(), "Base ONNX Runtime native library is not loadable — verify libonnxruntime.so is present and built for linux-x64 (try --runtime linux-x64)");
        Skip.IfNot(NetworkIsAvailable(), "Network is not available");

        var imageBytes = await DownloadImageAsync("https://ultralytics.com/images/bus.jpg");
        var engine = CreateEngine(useGpu: true);

        var results = await engine.PredictAsync(imageBytes);

        Assert.NotNull(results);
        Assert.NotEmpty(results);
        Assert.Contains(results, r => r.Label is "person" or "bus" or "car");
    }

    /// <summary>
    /// Validates that inference runs with the MIGraphX execution provider active when real AMD GPU
    /// hardware is available (<c>/dev/kfd</c> present) and all required runtimes (HSA, HIP, MIGraphX)
    /// are installed. Fails if the session falls back to CPU despite full ROCm stack being present.
    /// </summary>
    [SkippableFact]
    public async Task PredictAsync_OnLinuxX64WithAmdGpu_ShouldUseMIGraphXProvider()
    {
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "Test requires Linux x64 platform");
        Skip.IfNot(IsRocmRuntimeUsableOnHost(), "No usable AMD GPU on this host (no /dev/kfd or libhsa-runtime64.so unreachable) — requires self-hosted runner with ROCm installed");
        Skip.IfNot(IsMIGraphXRuntimeUsableOnHost(), "MIGraphX runtime not fully installed (missing libamdhip64.so.7 or libmigraphx_c.so.3) — requires self-hosted runner with full ROCm/MIGraphX stack");
        Skip.IfNot(IsMIGraphXLibPresent(), "libonnxruntime_providers_migraphx.so not found — build with --runtime linux-x64 to copy native assets");
        Skip.IfNot(IsBaseNativeOnnxRuntimeLoadable(), "Base ONNX Runtime native library is not loadable — verify libonnxruntime.so is present and built for linux-x64 (try --runtime linux-x64)");
        Skip.IfNot(NetworkIsAvailable(), "Network is not available");

        var imageBytes = await DownloadImageAsync("https://ultralytics.com/images/bus.jpg");
        var engine = CreateEngine(useGpu: true);

        var results = await engine.PredictAsync(imageBytes);

        Assert.True(engine.IsRocmSessionActive,
            "Expected MIGraphX execution provider to be active — AMD GPU present but MIGraphX session did not initialize. Check ROCm driver installation.");
        Assert.NotNull(results);
        Assert.NotEmpty(results);
        Assert.Contains(results, r => r.Label is "person" or "bus" or "car");
    }
}
