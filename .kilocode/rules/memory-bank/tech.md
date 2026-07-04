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
- GPU acceleration is platform-dependent: MIGraphX on Linux x64, DirectML on Windows, CoreML on macOS, CPU fallback elsewhere. The provider is auto-selected with a granular `FallbackReason` if it can't activate.
- On Linux x64, MIGraphX requires `libamdhip64.so.7` (HIP 7.x) + `libmigraphx_c.so.3` + `/dev/kfd`. Debian's `libamdhip64.so.5` (HIP 5.7) is ABI-incompatible and must NOT be used — install from the AMD ROCm apt repo.
- Integration tests can be skipped when external model/image URLs are unavailable or the ONNX Runtime native library is absent.

## Dependencies

- `Microsoft.ML.OnnxRuntime` 1.24.1 (managed bindings must match the native lib version).
- `Microsoft.ML.OnnxRuntime.Extensions` 0.14.0
- `Microsoft.ML.OnnxRuntime.DirectML` 1.23.0 (Windows RID only)
- `InferenceEngine.ROCm.Runtime.linux-x64` 1.24.1.36 (MIGraphX-enabled native libs, Linux x64 only)
- `Microsoft.Extensions.Logging.Abstractions`
- `SkiaSharp` (Examples only)
- `xunit`, `xunit.runner.visualstudio`, `Xunit.SkippableFact`, `Microsoft.NET.Test.Sdk`, `coverlet.collector`

## Packaging & publishing

- Package id: `InferenceEngine.Core`; default version `1.1.0-dev` in csproj, overridden in CI.
- CI versioning (`VERSION_PREFIX=1.0` repo variable + `github.run_number`): `development` → `<prefix>-dev.<run>` (e.g. `1.0.0-dev.11`), `staging` → `-rc`, `release` → `<prefix>.<run>`.
- Publish NuGet workflow (`on: push: [development, staging, release]`): GitHub Packages (all three) + NuGet.org (release only, needs `nuget-org` env + `NUGET_API_KEY`).
- GitHub Packages feed: `https://nuget.pkg.github.com/intel-agency/index.json` (consumers need a PAT with `read:packages`).

## Tool usage patterns

- Subclass `BaseInferenceEngine<TInput, TOutput>` and override preprocess and postprocess hooks.
- Use `LoadAsync` during startup to initialize session metadata and optional warmup.
- Use `PredictAsync` for runtime inference and rely on built-in disposal semantics.
- Keep inference provider selection centralized in the base engine; avoid provider branching in child engines.

