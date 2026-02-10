using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using InferenceEngine.Core;

namespace InferenceEngine.Core.Examples
{
    /// <summary>
    /// Specialized Inference Engine for Face/Object Detection (YOLOv8 based).
    /// Adapts the generic BaseInferenceEngine to handle Computer Vision specific tasks
    /// like image resizing, aspect ratio scaling, and coordinate mapping.
    /// </summary>
    public class FaceDetectionEngine : BaseInferenceEngine<byte[], List<FaceDetectionEngine.PredictionResult>>
    {
        // Configuration
        private const float ConfidenceThreshold = 0.45f;
        private const float IouThreshold = 0.5f;
        
        // Diagnostics
        private static readonly bool BboxDiagEnabled =
            string.Equals(Environment.GetEnvironmentVariable("INFERENCEAPP_BBOX_DIAG"), "1", StringComparison.OrdinalIgnoreCase);
        private const float CoordSpaceToleranceMargin = 1.15f;

        private readonly string[] _labels =
        [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ];

        public FaceDetectionEngine(InferenceEngineOptions options, ILogger<FaceDetectionEngine> logger) 
            : base(options, logger)
        {
        }

        // --- Pipeline Implementation ---

        /// <summary>
        /// Preprocesses the image and returns the tensor + scaling context.
        /// </summary>
        protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(byte[] imageBytes)
        {
            // 1. Decode
            using var sourceImage = SKBitmap.Decode(imageBytes) 
                ?? throw new Exception("Failed to decode input image bytes");

            // 2. Get Dimensions (Metadata is auto-populated by BaseInferenceEngine)
            // Dimensions are usually [-1, 3, Height, Width]
            int modelH = InputMetadata![PrimaryInputName!].Dimensions[2];
            int modelW = InputMetadata![PrimaryInputName!].Dimensions[3];

            // 3. Resize
            var resized = sourceImage.Resize(new SKImageInfo(modelW, modelH), SKSamplingOptions.Default);

            // 4. Calculate Scale Factors (Context)
            var context = new ImageScaleContext
            {
                ScaleX = (float)sourceImage.Width / modelW,
                ScaleY = (float)sourceImage.Height / modelH,
                SourceWidth = sourceImage.Width,
                SourceHeight = sourceImage.Height
            };

            // 5. To Tensor (NCHW)
            var tensor = new DenseTensor<float>(new[] { 1, 3, modelH, modelW });
            using var converted = resized.Copy(SKColorType.Rgba8888);
            var pixels = converted.Bytes;
            var bytesPerPixel = converted.BytesPerPixel;

            Parallel.For(0, modelH, y =>
            {
                for (int x = 0; x < modelW; x++)
                {
                    int offset = (y * modelW + x) * bytesPerPixel;
                    tensor[0, 0, y, x] = pixels[offset] / 255.0f; // R
                    tensor[0, 1, y, x] = pixels[offset + 1] / 255.0f; // G
                    tensor[0, 2, y, x] = pixels[offset + 2] / 255.0f; // B
                }
            });

            return (new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(PrimaryInputName, tensor) }, context);
        }

        /// <summary>
        /// Postprocesses the output tensors using the scaling context to map boxes.
        /// </summary>
        protected override List<PredictionResult> PostProcessWithContext(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, 
            object? context)
        {
            var scaleCtx = context as ImageScaleContext 
                ?? throw new InvalidOperationException("Invalid context type passed to PostProcess");

            var outputTensor = output.First().AsTensor<float>();
            var predictions = new List<PredictionResult>();

            int channels = outputTensor.Dimensions[1];
            int anchors = outputTensor.Dimensions[2];

            // 1. Determine Coordinate Space (Heuristic)
            float maxCx = float.NegativeInfinity, maxCy = float.NegativeInfinity;
            float maxW = float.NegativeInfinity, maxH = float.NegativeInfinity;

            for (int i = 0; i < anchors; i++)
            {
                if (outputTensor[0, 0, i] > maxCx) maxCx = outputTensor[0, 0, i];
                if (outputTensor[0, 1, i] > maxCy) maxCy = outputTensor[0, 1, i];
                if (outputTensor[0, 2, i] > maxW) maxW = outputTensor[0, 2, i];
                if (outputTensor[0, 3, i] > maxH) maxH = outputTensor[0, 3, i];
            }

            // Using Dimensions from Metadata for accurate space inference
            int modelH = InputMetadata![PrimaryInputName!].Dimensions[2];
            int modelW = InputMetadata![PrimaryInputName!].Dimensions[3];
            
            var coordSpace = InferBoxCoordSpace(maxCx, maxCy, maxW, maxH, scaleCtx.SourceWidth, scaleCtx.SourceHeight, modelW, modelH);

            // 2. Filter & Decode
            for (int i = 0; i < anchors; i++)
            {
                // Find max class score
                float maxScore = 0;
                int maxClassId = -1;

                for (int c = 4; c < channels; c++)
                {
                    float score = outputTensor[0, c, i];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        maxClassId = c - 4;
                    }
                }

                if (maxScore < ConfidenceThreshold) continue;

                // Extract Box
                float cx = outputTensor[0, 0, i];
                float cy = outputTensor[0, 1, i];
                float w = outputTensor[0, 2, i];
                float h = outputTensor[0, 3, i];

                // Center -> Top-Left
                float x = cx - (w / 2);
                float y = cy - (h / 2);

                // Decode
                var (px, py, pw, ph) = DecodeToSourcePixels(
                    coordSpace, x, y, w, h, 
                    scaleCtx.ScaleX, scaleCtx.ScaleY, 
                    scaleCtx.SourceWidth, scaleCtx.SourceHeight);

                predictions.Add(new PredictionResult
                {
                    Label = maxClassId >= 0 && maxClassId < _labels.Length ? _labels[maxClassId] : "Unknown",
                    Confidence = maxScore,
                    X = px, Y = py, Width = pw, Height = ph
                });
            }

            return ApplyNMS(predictions);
        }

        // --- DTOs & Helpers ---

        private class ImageScaleContext
        {
            public float ScaleX { get; set; }
            public float ScaleY { get; set; }
            public int SourceWidth { get; set; }
            public int SourceHeight { get; set; }
        }

        public class PredictionResult
        {
            public string Label { get; set; } = string.Empty;
            public float Confidence { get; set; }
            public float X { get; set; }
            public float Y { get; set; }
            public float Width { get; set; }
            public float Height { get; set; }
        }

        public enum BoxCoordSpace { Normalized01, ModelInputPixels, SourcePixels, Unknown }

        public BoxCoordSpace InferBoxCoordSpace(
            float maxCx, float maxCy, float maxW, float maxH, 
            int srcW, int srcH, int modelW, int modelH)
        {
            float maxAll = Math.Max(Math.Max(maxCx, maxCy), Math.Max(maxW, maxH));
            
            if (maxAll <= CoordSpaceToleranceMargin) return BoxCoordSpace.Normalized01;
            
            float modelMax = Math.Max(modelW, modelH);
            if (maxAll <= modelMax * CoordSpaceToleranceMargin) return BoxCoordSpace.ModelInputPixels;
            
            float srcMax = Math.Max(srcW, srcH);
            if (maxAll <= srcMax * CoordSpaceToleranceMargin) return BoxCoordSpace.SourcePixels;

            return BoxCoordSpace.Unknown;
        }

        public static (float X, float Y, float W, float H) DecodeToSourcePixels(
            BoxCoordSpace space, float x, float y, float w, float h, float sX, float sY, int srcW, int srcH)
        {
            return space switch
            {
                BoxCoordSpace.Normalized01 => (x * srcW, y * srcH, w * srcW, h * srcH),
                BoxCoordSpace.ModelInputPixels => (x * sX, y * sY, w * sX, h * sY),
                BoxCoordSpace.SourcePixels => (x, y, w, h),
                _ => throw new InvalidOperationException($"Unknown coordinate space detected.")
            };
        }

        protected List<PredictionResult> ApplyNMS(List<PredictionResult> boxes)
        {
            var sorted = boxes.OrderByDescending(b => b.Confidence).ToList();
            var selected = new List<PredictionResult>();

            while (sorted.Count > 0)
            {
                var best = sorted[0];
                selected.Add(best);
                sorted.RemoveAt(0);

                for (int i = sorted.Count - 1; i >= 0; i--)
                {
                    if (CalculateIoU(best, sorted[i]) > IouThreshold) sorted.RemoveAt(i);
                }
            }
            return selected;
        }

        protected float CalculateIoU(PredictionResult boxA, PredictionResult boxB)
        {
            float xA = Math.Max(boxA.X, boxB.X);
            float yA = Math.Max(boxA.Y, boxB.Y);
            float xB = Math.Min(boxA.X + boxA.Width, boxB.X + boxB.Width);
            float yB = Math.Min(boxA.Y + boxA.Height, boxB.Y + boxB.Height);

            float interArea = Math.Max(0, xB - xA) * Math.Max(0, yB - yA);
            float unionArea = (boxA.Width * boxA.Height) + (boxB.Width * boxB.Height) - interArea;

            return interArea / unionArea;
        }
    }
}
