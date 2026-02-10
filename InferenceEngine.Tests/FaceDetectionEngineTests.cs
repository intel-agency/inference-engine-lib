using System.Net;
using InferenceEngine.Core;
using InferenceEngine.Core.Examples;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace InferenceEngine.Tests;

public class FaceDetectionEngineTests
{
    private static readonly HttpClient Http = new();

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

    [Fact]
    public async Task LoadAsync_WithDefaultProvider_ShouldPopulateMetadata()
    {
        var engine = CreateEngine();

        await engine.LoadAsync();

        Assert.NotNull(engine.InputMetadata);
        Assert.NotNull(engine.OutputMetadata);
        Assert.False(string.IsNullOrWhiteSpace(engine.PrimaryInputName));
    }

    [Fact]
    public async Task PredictAsync_WithWebImage_ShouldReturnSomePredictions()
    {
        var imageBytes = await DownloadImageAsync("https://ultralytics.com/images/bus.jpg");
        var engine = CreateEngine();

        var results = await engine.PredictAsync(imageBytes);

        Assert.NotNull(results);
        Assert.NotEmpty(results);
    }

    [Fact]
    public async Task PredictAsync_WithBusImage_ShouldDetectAtLeastOnePersonOrBusOrCar()
    {
        var imageBytes = await DownloadImageAsync("https://ultralytics.com/images/bus.jpg");
        var engine = CreateEngine();

        var results = await engine.PredictAsync(imageBytes);

        Assert.Contains(results, r => r.Label is "person" or "bus" or "car");
    }
}
