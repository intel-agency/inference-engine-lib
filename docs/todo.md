# TODO

## Implement

### IEL3: Refactor project/solution structure to be .NET idiomatic

**Summary:** Refactor the solution into an idiomatic .NET layout with clearly separated core, test, and example projects, and consistent project configuration.

**Goals:**
- Align folder and project structure with common .NET conventions (for example, separating source, tests, and samples).
- Ensure each project has a focused responsibility (core library, examples, tooling/benchmarks, etc.).
- Standardize project naming, target frameworks, nullable/implicit usings, and shared build settings across projects.
- Simplify the solution by organizing related projects together and removing obsolete or redundant entries.

## Implemented

### IEL1: Move FaceDetectionEngine into its own project ✅

**Summary:** Created `InferenceEngine.Examples` project and moved `FaceDetectionEngine` from `InferenceEngine.Core/Examples/` to the new project.

**Changes:**
- Created `InferenceEngine.Examples/InferenceEngine.Examples.csproj` (net10.0 library)
- Moved `FaceDetectionEngine.cs` → `InferenceEngine.Examples/`
- Changed namespace `InferenceEngine.Core.Examples` → `InferenceEngine.Examples`
- Added project reference in solution (`inference-engine-lib.slnx`)
- Updated `InferenceEngine.Tests` to reference new project and updated usings
- Removed `SkiaSharp` dependency from `InferenceEngine.Core` (moved to Examples)
- Removed empty `Examples/` directory

**Result:** Core library is now lighter (no SkiaSharp dependency), FaceDetectionEngine is in its own reusable library. All 17 tests pass.

### IEL2: Add default registry of download URL locations for required models ✅

**Summary:** Created `ModelRegistry` class that loads model download URLs from a JSON file, allowing engines to specify only a model name for commonly known models.

**Changes:**
- Added `ModelRegistry.cs` - loads and queries model URLs from JSON registry
- Added `models.json` - default registry with YOLO models (yolo11n/s/m, yolov8n) from GitHub and Hugging Face
- Modified `BaseInferenceEngine` to auto-lookup URLs from registry when `ModelDownloadUrls` is not specified
- Updated `FaceDetectionEngine` to demonstrate registry usage (removed explicit URLs)
- Updated `.csproj` to copy `models.json` to output and include in NuGet package

**Benefits:**
- Client extensions only need to specify `ModelName` for registered models
- URLs can be updated without code changes
- Multiple mirror URLs per model for reliability
- Easy to add new models to the registry

**Result:** FaceDetectionEngine now relies on the registry for yolo11n.onnx URLs. All 17 tests pass.

**Test Coverage Improvements:** Added comprehensive test suite (62 total tests):
- ModelRegistryTests (17 tests) - JSON loading, lookups, edge cases
- LocalModelProviderTests (8 tests) - file operations, error handling  
- DownloadingModelProviderTests (8 tests) - caching, failover, error paths
- BaseInferenceEngineTests (15 tests) - registry integration, options, lifecycle
- Existing tests: 14 tests (Basic, CoordinateSpace, NMS, FaceDetection)

**Total: 62 tests, all passing**
