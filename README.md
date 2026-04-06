# InferenceEngine

**A cross-platform .NET library for AI inference with ONNX Runtime.**

[![Build & Test](https://github.com/intel-agency/inference-engine-lib/actions/workflows/build-test.yml/badge.svg)](https://github.com/intel-agency/inference-engine-lib/actions/workflows/build-test.yml)
[![NuGet](https://img.shields.io/nuget/v/InferenceEngine.Core?label=NuGet)](https://www.nuget.org/packages/InferenceEngine.Core)
[![License: AGPL v3+](https://img.shields.io/badge/License-AGPL_v3+-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![.NET 10](https://img.shields.io/badge/.NET-10.0-purple)](https://dotnet.microsoft.com/)

Hardware-accelerated ONNX inference behind a generic `IInferenceEngine<TInput, TOutput>` contract — build vision agents, NLP pipelines, or any ML-powered service with a unified API.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Model Providers](#model-providers)
- [Hardware Acceleration](#hardware-acceleration)
- [Examples](#examples)
- [Configuration](#configuration)
- [Testing](#testing)
- [CI/CD](#cicd)
- [AI Agents Integration](#ai-agents-integration)
- [Contributing](#contributing)
- [License](#license)

## Features

| Capability | Description |
| --- | --- |
| **Generic Interface** | `IInferenceEngine<TInput, TOutput>` — plug in any input/output type |
| **Hardware Acceleration** | Auto-detects DirectML (Windows), CoreML (macOS), CPU fallback (Linux) |
| **Cross-Platform** | Targets `win-x64`, `win-arm64`, `linux-x64`, `linux-arm64`, `osx-x64`, `osx-arm64` |
| **Model Registry** | JSON-based registry with automatic download and local caching |
| **Context Passing** | Share preprocessing state (e.g. scale factors) with postprocessing via a context object |
| **Warmup on Load** | Eliminates first-run latency by compiling GPU shaders / graphs at startup |
| **Resource Safety** | Full `IDisposable` lifecycle — sessions, tensors, and native memory are cleaned up |

## Quick Start

### Install

```bash
dotnet add package InferenceEngine.Core
```

### Minimal Usage

```csharp
using InferenceEngine.Core;

var options = new InferenceEngineOptions
{
    ModelPath = "path/to/model.onnx",   // or let the engine download from the registry
    UseGpuAcceleration = true,
    WarmupOnLoad = true
};

var engine = new YourCustomEngine(options, logger);
await engine.LoadAsync();

var result = await engine.PredictAsync(inputData);
```

> **Tip:** If you omit `ModelPath`, the engine resolves the model through its `ModelName` property and the built-in `ModelRegistry`, downloading and caching the file automatically.

## Architecture

```text
InferenceEngine.Core/
├── IInferenceEngine<TInput, TOutput>   # Generic contract (Load + Predict)
├── BaseInferenceEngine<TInput, TOutput># Abstract base — ONNX Runtime session, HW accel, warmup
├── InferenceEngineOptions              # Configuration (GPU, device ID, warmup, model provider)
└── ModelProviders/
    ├── IModelProvider                  # Abstraction for model acquisition
    ├── LocalModelProvider              # Load from a local file path
    ├── DownloadingModelProvider         # Download with multi-URL fallback + local cache
    └── ModelRegistry                   # JSON lookup of model names → download URLs

InferenceEngine.Examples/
└── FaceDetectionEngine                 # YOLO11-based object detection (SkiaSharp preprocessing)

InferenceEngine.Tests/
├── BaseInferenceEngineTests
├── FaceDetectionEngineTests
├── ModelRegistryTests
├── DownloadingModelProviderTests
├── LocalModelProviderTests
├── CoordinateSpaceTests
└── NonMaximumSuppressionTests
```

### Implementing a Custom Engine

Subclass `BaseInferenceEngine<TInput, TOutput>` and implement two methods:

```csharp
public class CustomEngine : BaseInferenceEngine<byte[], List<Prediction>>
{
    protected override string ModelName => "my-model.onnx";

    protected override (List<NamedOnnxValue> InputTensors, object? Context)
        PreProcessWithContext(byte[] input)
    {
        // Convert raw input into ONNX tensors.
        // Optionally return a context object (e.g. scaling factors).
    }

    protected override List<Prediction> PostProcessWithContext(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output,
        object? context)
    {
        // Decode ONNX output tensors into your domain type.
        // Use the context from preprocessing if needed.
    }
}
```

## Model Providers

The library decouples **model acquisition** from **inference** through the `IModelProvider` interface.

| Provider | Behavior |
| --- | --- |
| `LocalModelProvider` | Resolves a path on disk; throws if the file is missing |
| `DownloadingModelProvider` | Tries a prioritised list of URLs, caches the result locally |
| `ModelRegistry` | Maps a model name (e.g. `yolo11n.onnx`) to download URLs via `models.json` |

**Resolution order** when an engine loads:

1. Explicit `InferenceEngineOptions.ModelProvider` (if set)
2. Engine's `ModelDownloadUrls` override
3. `ModelRegistry` lookup by `ModelName`

### Extending the Registry

Add entries to [`InferenceEngine.Core/models.json`](InferenceEngine.Core/models.json):

```json
{
  "name": "my-custom-model.onnx",
  "description": "Custom model for my agent",
  "source": "My Org",
  "urls": [
    "https://example.com/models/my-custom-model.onnx"
  ]
}
```

## Hardware Acceleration

| Platform | Execution Provider | Notes |
| --- | --- | --- |
| **Windows** | DirectML | GPU acceleration via `AppendExecutionProvider_DML` |
| **macOS** | CoreML | Neural Engine / GPU via `AppendExecutionProvider_CoreML` |
| **Linux x64** | ROCm / CPU fallback | When `UseGpuAcceleration` is enabled, the engine attempts ROCm first; requires ROCm-enabled ONNX Runtime native assets at deployment time |
| **Linux (other architectures)** | CPU | Uses the default CPU provider |

- Set `UseGpuAcceleration = false` to force CPU on any platform.
- On Linux x64, GPU acceleration requires ROCm to be available on the host and the application to ship with ROCm-enabled ONNX Runtime native dependencies (included in the `InferenceEngine.Core` ROCm NuGet package).
- If the preferred provider fails to load, the engine falls back to CPU automatically.

## Examples

### Face / Object Detection

[`InferenceEngine.Examples/FaceDetectionEngine.cs`](InferenceEngine.Examples/FaceDetectionEngine.cs) demonstrates a full computer-vision pipeline:

- **Image preprocessing** — decode, resize, normalize to NCHW tensor with SkiaSharp
- **YOLO11 object detection** — 80-class COCO detection
- **Coordinate space inference** — automatically determines if model outputs normalized, model-pixel, or source-pixel coordinates
- **Non-maximum suppression** — IoU-based duplicate filtering
- **Automatic model download** — resolves `yolo11n.onnx` from the built-in registry

```csharp
var engine = new FaceDetectionEngine(
    new InferenceEngineOptions { UseGpuAcceleration = true },
    logger);

await engine.LoadAsync();
var detections = await engine.PredictAsync(imageBytes);

foreach (var det in detections)
    Console.WriteLine($"{det.Label} ({det.Confidence:P0}) at [{det.X}, {det.Y}, {det.Width}, {det.Height}]");
```

## Configuration

### `InferenceEngineOptions`

| Property | Type | Default | Description |
| --- | --- | --- | --- |
| `ModelProvider` | `IModelProvider?` | `null` | Explicit model provider; overrides all other resolution |
| `ModelPath` | `string?` | `null` | Convenience shortcut — sets a `LocalModelProvider` internally |
| `UseGpuAcceleration` | `bool` | `true` | Enable platform-specific GPU execution providers |
| `DeviceId` | `int` | `0` | GPU device index for multi-GPU systems |
| `WarmupOnLoad` | `bool` | `true` | Run a dummy inference during `LoadAsync` to pre-compile GPU graphs |

## Testing

The test suite uses **xUnit** with code coverage via **Coverlet**.

```bash
# Run all tests
dotnet test

# Run with coverage
dotnet test --collect:"XPlat Code Coverage"
```

> Integration tests that require a GPU or model download (e.g. `FaceDetectionEngineTests`) are skipped on Linux/macOS CI by default.

## CI/CD

| Workflow | Trigger | Description |
| --- | --- | --- |
| **Build & Test** | Push / PR to `main`, `development` | Builds and tests on `ubuntu`, `windows`, `macos`; uploads coverage reports |
| **Publish NuGet** | Push to `release` | Packs and publishes to **GitHub Packages** and **NuGet.org** |

Both workflows target **.NET 10** and use a reusable composite action at `.github/actions/build-and-test`.

## AI Agents Integration

This library is purpose-built for AI agent architectures. See [**AGENTS.md**](AGENTS.md) for:

- Vision agent and multi-modal agent integration patterns
- Custom engine implementation guidelines
- Agent lifecycle management (init → predict → dispose)
- Performance tuning for production workloads

## Contributing

1. Fork the repository
2. Create a feature branch from `development` (`feature/<descriptive-name>`)
3. Open a PR targeting `development`
4. Ensure CI passes and resolve all review comments

See [AGENTS.md § Branch Strategy](AGENTS.md#repository-branch-strategy) for the full branching model.

## Dependencies

| Package | Purpose |
| --- | --- |
| `Microsoft.ML.OnnxRuntime` | Core ONNX inference runtime |
| `Microsoft.ML.OnnxRuntime.Extensions` | Custom operator support |
| `Microsoft.ML.OnnxRuntime.DirectML` | Windows GPU acceleration (conditional) |
| `Microsoft.Extensions.Logging.Abstractions` | Logging contract |
| `SkiaSharp` | Image processing (Examples project) |

## License

**AGPL-3.0-or-later** — Copyright © 2026 Artificial Intelligence Agency. All rights reserved.
