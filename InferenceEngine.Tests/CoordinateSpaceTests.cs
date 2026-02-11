using InferenceEngine.Examples;

namespace InferenceEngine.Tests;

public class CoordinateSpaceTests
{
    [Fact]
    public void InferBoxCoordSpace_WithNormalizedValues_ShouldReturnNormalized01()
    {
        var engine = new FaceDetectionEngine(
            new InferenceEngine.Core.InferenceEngineOptions { ModelPath = "dummy.onnx", UseGpuAcceleration = false, WarmupOnLoad = false },
            Microsoft.Extensions.Logging.Abstractions.NullLogger<FaceDetectionEngine>.Instance);

        var space = engine.InferBoxCoordSpace(
            maxCx: 0.9f,
            maxCy: 0.8f,
            maxW: 0.7f,
            maxH: 0.6f,
            srcW: 1920,
            srcH: 1080,
            modelW: 640,
            modelH: 640);

        Assert.Equal(FaceDetectionEngine.BoxCoordSpace.Normalized01, space);
    }

    [Fact]
    public void InferBoxCoordSpace_WithModelPixelValues_ShouldReturnModelInputPixels()
    {
        var engine = new FaceDetectionEngine(
            new InferenceEngine.Core.InferenceEngineOptions { ModelPath = "dummy.onnx", UseGpuAcceleration = false, WarmupOnLoad = false },
            Microsoft.Extensions.Logging.Abstractions.NullLogger<FaceDetectionEngine>.Instance);

        var space = engine.InferBoxCoordSpace(
            maxCx: 500f,
            maxCy: 400f,
            maxW: 300f,
            maxH: 200f,
            srcW: 1920,
            srcH: 1080,
            modelW: 640,
            modelH: 640);

        Assert.Equal(FaceDetectionEngine.BoxCoordSpace.ModelInputPixels, space);
    }

    [Fact]
    public void InferBoxCoordSpace_WithSourcePixelValues_ShouldReturnSourcePixels()
    {
        var engine = new FaceDetectionEngine(
            new InferenceEngine.Core.InferenceEngineOptions { ModelPath = "dummy.onnx", UseGpuAcceleration = false, WarmupOnLoad = false },
            Microsoft.Extensions.Logging.Abstractions.NullLogger<FaceDetectionEngine>.Instance);

        var space = engine.InferBoxCoordSpace(
            maxCx: 1500f,
            maxCy: 900f,
            maxW: 800f,
            maxH: 600f,
            srcW: 1920,
            srcH: 1080,
            modelW: 640,
            modelH: 640);

        Assert.Equal(FaceDetectionEngine.BoxCoordSpace.SourcePixels, space);
    }

    [Fact]
    public void DecodeToSourcePixels_Normalized01_ShouldScaleBySourceDimensions()
    {
        var (x, y, w, h) = FaceDetectionEngine.DecodeToSourcePixels(
            FaceDetectionEngine.BoxCoordSpace.Normalized01,
            x: 0.1f,
            y: 0.2f,
            w: 0.3f,
            h: 0.4f,
            sX: 3f,
            sY: 2f,
            srcW: 1000,
            srcH: 500);

        Assert.Equal(100f, x, 3);
        Assert.Equal(100f, y, 3);
        Assert.Equal(300f, w, 3);
        Assert.Equal(200f, h, 3);
    }

    [Fact]
    public void DecodeToSourcePixels_ModelInputPixels_ShouldScaleByScaleFactors()
    {
        var (x, y, w, h) = FaceDetectionEngine.DecodeToSourcePixels(
            FaceDetectionEngine.BoxCoordSpace.ModelInputPixels,
            x: 10f,
            y: 20f,
            w: 30f,
            h: 40f,
            sX: 2f,
            sY: 3f,
            srcW: 1000,
            srcH: 500);

        Assert.Equal(20f, x, 3);
        Assert.Equal(60f, y, 3);
        Assert.Equal(60f, w, 3);
        Assert.Equal(120f, h, 3);
    }

    [Fact]
    public void DecodeToSourcePixels_SourcePixels_ShouldReturnIdentity()
    {
        var (x, y, w, h) = FaceDetectionEngine.DecodeToSourcePixels(
            FaceDetectionEngine.BoxCoordSpace.SourcePixels,
            x: 10f,
            y: 20f,
            w: 30f,
            h: 40f,
            sX: 2f,
            sY: 3f,
            srcW: 1000,
            srcH: 500);

        Assert.Equal(10f, x, 3);
        Assert.Equal(20f, y, 3);
        Assert.Equal(30f, w, 3);
        Assert.Equal(40f, h, 3);
    }

    [Fact]
    public void DecodeToSourcePixels_Unknown_ShouldThrow()
    {
        Assert.Throws<InvalidOperationException>(() =>
            FaceDetectionEngine.DecodeToSourcePixels(
                FaceDetectionEngine.BoxCoordSpace.Unknown,
                x: 0,
                y: 0,
                w: 0,
                h: 0,
                sX: 1,
                sY: 1,
                srcW: 1,
                srcH: 1));
    }
}
