using System.Net;
using System.Runtime.InteropServices;
using InferenceEngine.Core;
using InferenceEngine.Examples;
using Microsoft.Extensions.Logging.Abstractions;
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
        var imageBytes = await DownloadImageAsync("https://ultralytics.com/images/bus.jpg");
        var engine = CreateEngine();

        var results = await engine.PredictAsync(imageBytes);

        Assert.Contains(results, r => r.Label is "person" or "bus" or "car");
    }

    // --- ROCm GPU Inference Tests ---

    /// <summary>
    /// Returns true when <c>libonnxruntime_providers_rocm.so</c> is present in the output
    /// directory, indicating the <c>InferenceEngine.ROCm.Runtime.linux-x64</c> NuGet package
    /// has been restored and its native assets were copied to the build output.
    /// </summary>
    private static bool IsRocmLibPresent()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "libonnxruntime_providers_rocm.so"),
            Path.Combine(AppContext.BaseDirectory, "runtimes", "linux-x64", "native", "libonnxruntime_providers_rocm.so"),
        };
        return candidates.Any(File.Exists);
    }

    /// <summary>
    /// Returns true when an AMD GPU is accessible via the ROCm kernel driver
    /// (<c>/dev/kfd</c> is present on the host).
    /// </summary>
    private static bool IsAmdGpuAvailable() =>
        RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && File.Exists("/dev/kfd");

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
        Skip.IfNot(IsRocmLibPresent(), "libonnxruntime_providers_rocm.so not found — InferenceEngine.ROCm.Runtime.linux-x64 package assets must be present");
        Skip.IfNot(NetworkIsAvailable(), "Network is not available");

        var imageBytes = await DownloadImageAsync("https://ultralytics.com/images/bus.jpg");
        var engine = CreateEngine(useGpu: true);

        var results = await engine.PredictAsync(imageBytes);

        Assert.NotNull(results);
        Assert.NotEmpty(results);
        Assert.Contains(results, r => r.Label is "person" or "bus" or "car");
    }

    /// <summary>
    /// Validates that inference runs with the ROCm execution provider active when real AMD GPU
    /// hardware is available (<c>/dev/kfd</c> present). Fails if the session falls back to CPU
    /// despite ROCm hardware being present.
    /// </summary>
    [SkippableFact]
    public async Task PredictAsync_OnLinuxX64WithAmdGpu_ShouldUseRocmProvider()
    {
        Skip.IfNot(
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && RuntimeInformation.ProcessArchitecture == Architecture.X64,
            "Test requires Linux x64 platform");
        Skip.IfNot(IsAmdGpuAvailable(), "No AMD GPU detected (/dev/kfd not present) — requires self-hosted runner with AMD GPU");
        Skip.IfNot(IsRocmLibPresent(), "libonnxruntime_providers_rocm.so not found — build with --runtime linux-x64 to copy native assets");
        Skip.IfNot(NetworkIsAvailable(), "Network is not available");

        var imageBytes = await DownloadImageAsync("https://ultralytics.com/images/bus.jpg");
        var engine = CreateEngine(useGpu: true);

        var results = await engine.PredictAsync(imageBytes);

        Assert.True(engine.IsRocmSessionActive,
            "Expected ROCm execution provider to be active — AMD GPU present but ROCm session did not initialize. Check ROCm driver installation.");
        Assert.NotNull(results);
        Assert.NotEmpty(results);
        Assert.Contains(results, r => r.Label is "person" or "bus" or "car");
    }
}
