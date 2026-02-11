# Architecture

## System architecture

- `InferenceEngine.Core` contains the reusable inference abstraction and ONNX Runtime execution pipeline.
- `InferenceEngine.Tests` validates core behavior, coordinate logic, NMS behavior, and integration-level model inference paths.
- `FaceDetectionEngine` in examples demonstrates a concrete computer-vision specialization built on the core base class.

## Source code paths

- `InferenceEngine.Core/IInferenceEngine.cs` defines the generic inference contract.
- `InferenceEngine.Core/BaseInferenceEngine.cs` implements load, provider selection, warmup, run, and disposal lifecycle.
- `InferenceEngine.Core/InferenceEngineOptions.cs` defines runtime options for model path and acceleration behavior.
- `InferenceEngine.Core/Examples/FaceDetectionEngine.cs` implements image preprocessing, output decoding, coordinate handling, and NMS.
- `InferenceEngine.Tests/*.cs` covers unit, algorithm, and integration scenarios.

## Key technical decisions

- Use ONNX Runtime as the common inference backend.
- Expose a generic interface to support multiple input-output type pairs.
- Keep preprocessing and postprocessing abstract so domain engines own data transformation logic.
- Use runtime OS detection to select DirectML on Windows, CoreML on macOS, and CPU fallback elsewhere.

## Design patterns in use

- Template method pattern in `BaseInferenceEngine` for fixed pipeline with custom preprocess/postprocess hooks.
- Strategy-like execution provider selection based on platform and configuration.
- DTO-style prediction records for example engine outputs.

## Component relationships

- Consumers implement custom engines by inheriting from `BaseInferenceEngine<TInput, TOutput>`.
- `BaseInferenceEngine` depends on `InferenceEngineOptions` and ONNX Runtime session APIs.
- Example engine depends on SkiaSharp for image decode/resize and ONNX tensors for model inputs.
- Tests reference both core and example components to verify end-to-end behavior.

## Critical implementation paths

- `LoadAsync`: validates model path, creates session, captures metadata, and runs optional warmup.
- `PredictAsync`: preprocesses inputs, executes model run, postprocesses outputs, and disposes temporary tensors.
- Example postprocess path: infer coordinate space, decode boxes to source pixels, apply confidence filtering and NMS.

