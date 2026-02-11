# Inference Engine Library

A cross-platform .NET library for AI inference with ONNX Runtime, featuring hardware acceleration and a generic interface design for building AI agents.

## Features

- **Generic Interface**: `IInferenceEngine<TInput, TOutput>` supports any input/output types
- **Hardware Acceleration**: Automatic detection and use of DirectML (Windows) or CoreML (macOS)
- **Cross-Platform**: Supports Windows, macOS, and Linux
- **Resource Management**: Proper disposal and memory management
- **Context Passing**: Share state between preprocessing and postprocessing steps
- **Warmup Support**: Eliminates first-run latency through model warmup

## Quick Start

### Installation
```bash
dotnet add package InferenceEngine.Core
```

### Basic Usage
```csharp
var options = new InferenceEngineOptions
{
    ModelPath = "path/to/model.onnx",
    UseGpuAcceleration = true,
    WarmupOnLoad = true
};

var engine = new YourCustomEngine(options, logger);
await engine.LoadAsync();
var result = await engine.PredictAsync(inputData);
```

## Architecture

### Core Components

- **IInferenceEngine<TInput, TOutput>**: Generic contract for inference engines
- **BaseInferenceEngine**: Abstract base class with ONNX Runtime integration
- **InferenceEngineOptions**: Configuration class for engine setup

### Implementation Pattern

1. Inherit from `BaseInferenceEngine<TInput, TOutput>`
2. Implement `PreProcessWithContext()` for input preparation
3. Implement `PostProcessWithContext()` for output processing

```csharp
public class CustomEngine : BaseInferenceEngine<byte[], List<Prediction>>
{
    protected override (List<NamedOnnxValue>, object?) PreProcessWithContext(byte[] input)
    {
        // Convert input to ONNX tensors
        // Return tensors and optional context
    }
    
    protected override List<Prediction> PostProcessWithContext(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, 
        object? context)
    {
        // Process ONNX output tensors
        // Use context from preprocessing if needed
    }
}
```

## Hardware Acceleration

The library automatically detects and configures the best available hardware acceleration:

- **Windows**: DirectML for GPU acceleration
- **macOS**: CoreML for Neural Engine/GPU acceleration  
- **Linux**: CPU execution (with potential for CUDA in future)

Fallback to CPU occurs automatically if hardware acceleration fails.

## Examples

See `Examples/FaceDetectionEngine.cs` for a complete computer vision implementation featuring:
- Image preprocessing with SkiaSharp
- YOLOv8-style object detection
- Non-maximum suppression
- Coordinate space inference and mapping

## Configuration

### InferenceEngineOptions

```csharp
public class InferenceEngineOptions
{
    public string ModelPath { get; set; }           // Path to .onnx model
    public bool UseGpuAcceleration { get; set; }    // Enable hardware acceleration
    public int DeviceId { get; set; }               // GPU device ID (multi-GPU)
    public bool WarmupOnLoad { get; set; }          // Perform warmup inference
}
```

## AI Agents Integration

This library is designed for building AI agents. See [AGENTS.md](AGENTS.md) for detailed integration patterns and examples.

## Dependencies

### Core

- Microsoft.ML.OnnxRuntime
- Microsoft.ML.OnnxRuntime.Extensions
- Microsoft.Extensions.Logging.Abstractions
- SkiaSharp (for image processing examples)

### Platform-Specific (Hardware Acceleration)

- Microsoft.ML.OnnxRuntime.DirectML (Windows only)

## License

AGPL-3.0-or-later — see LICENSE file for details.
