# Product

## Why this project exists

Inference Engine Library provides a reusable inference foundation for .NET applications that need AI model execution without re-implementing model session lifecycle and platform-specific acceleration logic.

## Problems it solves

- Avoids duplicated inference plumbing across projects.
- Standardizes model load, warmup, inference, and cleanup flow.
- Reduces friction for building agent-style components on top of ONNX Runtime.
- Handles cross-platform acceleration selection with CPU fallback.

## How it should work

- Consumers implement domain-specific engines by inheriting from `BaseInferenceEngine<TInput, TOutput>`.
- Engines translate raw inputs into ONNX tensors in preprocess and map outputs in postprocess.
- `LoadAsync` initializes session metadata and optional warmup.
- `PredictAsync` runs preprocess, inference, and postprocess consistently.

## User experience goals

- Fast path from model file to working predictions.
- Clear extension points for custom pipelines.
- Predictable runtime behavior across Windows, macOS, and Linux.
- Reliable test coverage for core options and detection algorithms.

