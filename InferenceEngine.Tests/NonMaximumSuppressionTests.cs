using InferenceEngine.Core;
using InferenceEngine.Core.Examples;
using Microsoft.Extensions.Logging.Abstractions;

namespace InferenceEngine.Tests;

public class NonMaximumSuppressionTests
{
    private sealed class TestableFaceDetectionEngine : FaceDetectionEngine
    {
        public TestableFaceDetectionEngine() : base(
            new InferenceEngineOptions { ModelPath = "dummy.onnx", UseGpuAcceleration = false, WarmupOnLoad = false },
            NullLogger<FaceDetectionEngine>.Instance)
        {
        }

        public float IoU(PredictionResult a, PredictionResult b) => CalculateIoU(a, b);
        public List<PredictionResult> Nms(List<PredictionResult> boxes) => ApplyNMS(boxes);
    }

    [Fact]
    public void CalculateIoU_WithIdenticalBoxes_ShouldBeOne()
    {
        var engine = new TestableFaceDetectionEngine();

        var a = new FaceDetectionEngine.PredictionResult { X = 10, Y = 10, Width = 20, Height = 20, Confidence = 0.9f, Label = "person" };
        var b = new FaceDetectionEngine.PredictionResult { X = 10, Y = 10, Width = 20, Height = 20, Confidence = 0.8f, Label = "person" };

        var iou = engine.IoU(a, b);
        Assert.Equal(1f, iou, 3);
    }

    [Fact]
    public void CalculateIoU_WithNonOverlappingBoxes_ShouldBeZero()
    {
        var engine = new TestableFaceDetectionEngine();

        var a = new FaceDetectionEngine.PredictionResult { X = 0, Y = 0, Width = 10, Height = 10, Confidence = 0.9f, Label = "person" };
        var b = new FaceDetectionEngine.PredictionResult { X = 100, Y = 100, Width = 10, Height = 10, Confidence = 0.8f, Label = "person" };

        var iou = engine.IoU(a, b);
        Assert.Equal(0f, iou, 3);
    }

    [Fact]
    public void ApplyNMS_WithOverlappingBoxes_ShouldKeepHighestConfidence()
    {
        var engine = new TestableFaceDetectionEngine();

        var boxes = new List<FaceDetectionEngine.PredictionResult>
        {
            new() { X = 10, Y = 10, Width = 50, Height = 50, Confidence = 0.95f, Label = "person" },
            new() { X = 12, Y = 12, Width = 50, Height = 50, Confidence = 0.90f, Label = "person" },
            new() { X = 200, Y = 200, Width = 20, Height = 20, Confidence = 0.80f, Label = "car" }
        };

        var selected = engine.Nms(boxes);

        Assert.Contains(selected, b => b.Confidence == 0.95f);
        Assert.Contains(selected, b => b.Label == "car");
        Assert.DoesNotContain(selected, b => b.Confidence == 0.90f);
    }
}
