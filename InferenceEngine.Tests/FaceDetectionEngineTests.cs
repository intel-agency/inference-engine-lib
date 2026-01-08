using System.Net;
using InferenceEngine.Core;
using InferenceEngine.Core.Examples;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace InferenceEngine.Tests;

public class FaceDetectionEngineTests
{
    private static readonly HttpClient Http = new();

    private static async Task<byte[]> DownloadBytesAsync(string url)
    {
        using var resp = await Http.GetAsync(url);
        if (resp.StatusCode != HttpStatusCode.OK)
        {
            throw new InvalidOperationException($"Failed to download '{url}'. Status: {(int)resp.StatusCode} {resp.ReasonPhrase}");
        }
        return await resp.Content.ReadAsByteArrayAsync();
    }

    private static async Task<string?> TryEnsureModelAsync()
    {
        var dir = Path.Combine(AppContext.BaseDirectory, "test-assets");
        Directory.CreateDirectory(dir);

        var modelPath = Path.Combine(dir, "yolov8n.onnx");
        if (File.Exists(modelPath)) return modelPath;

        // Best-effort: external assets can move. If we can't download, tests will be skipped.
        var urls = new[]
        {
            // Ultralytics assets releases (versioned)
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx",
            "https://github.com/ultralytics/assets/releases/download/v8.5.0/yolov8n.onnx",
            "https://github.com/ultralytics/assets/releases/download/v8.6.0/yolov8n.onnx",
            // Fallback to ultralytics main repo raw (sometimes present)
            "https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/models/yolov8n.onnx",
            "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/yolov8n.onnx",
        };

        foreach (var url in urls)
        {
            try
            {
                var bytes = await DownloadBytesAsync(url);
                await File.WriteAllBytesAsync(modelPath, bytes);
                return modelPath;
            }
            catch
            {
                // swallow and try next
            }
        }

        return null;
    }

    private static async Task<byte[]?> TryDownloadImageAsync(string url)
    {
        try
        {
            return await DownloadBytesAsync(url);
        }
        catch
        {
            return null;
        }
    }

    [SkippableFact]
    public async Task LoadAsync_WithValidOnnxFile_ShouldPopulateMetadata()
    {
        var modelPath = await TryEnsureModelAsync();
        Skip.If(modelPath is null, "Skipping integration test: unable to download YOLOv8n ONNX model.");

        var engine = new FaceDetectionEngine(
            new InferenceEngineOptions
            {
                ModelPath = modelPath!,
                UseGpuAcceleration = false,
                WarmupOnLoad = false
            },
            NullLogger<FaceDetectionEngine>.Instance);

        await engine.LoadAsync();

        Assert.NotNull(engine.InputMetadata);
        Assert.NotNull(engine.OutputMetadata);
        Assert.False(string.IsNullOrWhiteSpace(engine.PrimaryInputName));
    }

    [SkippableFact]
    public async Task PredictAsync_WithWebImage_ShouldReturnSomePredictions()
    {
        var modelPath = await TryEnsureModelAsync();
        Skip.If(modelPath is null, "Skipping integration test: unable to download YOLOv8n ONNX model.");

        var imageBytes = await TryDownloadImageAsync("https://ultralytics.com/images/bus.jpg");
        Skip.If(imageBytes is null, "Skipping integration test: unable to download test image.");

        var engine = new FaceDetectionEngine(
            new InferenceEngineOptions
            {
                ModelPath = modelPath!,
                UseGpuAcceleration = false,
                WarmupOnLoad = false
            },
            NullLogger<FaceDetectionEngine>.Instance);

        var results = await engine.PredictAsync(imageBytes!);

        Assert.NotNull(results);
        Assert.NotEmpty(results);
    }

    [SkippableFact]
    public async Task PredictAsync_WithBusImage_ShouldDetectAtLeastOnePersonOrBusOrCar()
    {
        var modelPath = await TryEnsureModelAsync();
        Skip.If(modelPath is null, "Skipping integration test: unable to download YOLOv8n ONNX model.");

        var imageBytes = await TryDownloadImageAsync("https://ultralytics.com/images/bus.jpg");
        Skip.If(imageBytes is null, "Skipping integration test: unable to download test image.");

        var engine = new FaceDetectionEngine(
            new InferenceEngineOptions
            {
                ModelPath = modelPath!,
                UseGpuAcceleration = false,
                WarmupOnLoad = false
            },
            NullLogger<FaceDetectionEngine>.Instance);

        var results = await engine.PredictAsync(imageBytes!);

        Assert.Contains(results, r => r.Label is "person" or "bus" or "car");
    }
}
