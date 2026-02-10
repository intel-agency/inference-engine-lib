# Tech

## Technologies used

- .NET 10 class library and test projects.
- ONNX Runtime as the inference backend.
- SkiaSharp for image decode and resize in the vision example.
- xUnit with SkippableFact for unit and integration testing.

## Development setup

- Solution file `inference-engine-lib.slnx` with `InferenceEngine.Core` and `InferenceEngine.Tests`.
- Target framework is `net10.0` for both core and tests.
- Test project downloads model and sample images at runtime for integration scenarios.

## Technical constraints

- Model path must point to a valid `.onnx` file before load.
- GPU acceleration is platform-dependent: DirectML on Windows, CoreML on macOS, CPU fallback elsewhere.
- Integration tests can be skipped when external model/image URLs are unavailable.

## Dependencies

- `Microsoft.ML.OnnxRuntime`
- `Microsoft.ML.OnnxRuntime.Extensions`
- `Microsoft.ML.OnnxRuntime.DirectML`
- `Microsoft.Extensions.Logging.Abstractions`
- `SkiaSharp`
- `xunit`, `xunit.runner.visualstudio`, `Xunit.SkippableFact`, `Microsoft.NET.Test.Sdk`, `coverlet.collector`

## Tool usage patterns

- Subclass `BaseInferenceEngine<TInput, TOutput>` and override preprocess and postprocess hooks.
- Use `LoadAsync` during startup to initialize session metadata and optional warmup.
- Use `PredictAsync` for runtime inference and rely on built-in disposal semantics.
- Keep inference provider selection centralized in the base engine; avoid provider branching in child engines.

