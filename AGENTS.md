# AGENTS.md — AI Coding Agent Instructions

> Instructions for AI coding agents (GitHub Copilot, Codex, etc.) working in this repository.

---

## Project Overview

**InferenceEngine** is a cross-platform .NET 10 library that wraps ONNX Runtime behind a generic `IInferenceEngine<TInput, TOutput>` interface with automatic hardware acceleration (DirectML on Windows, CoreML on macOS, CPU on Linux).

### Solution Structure

| Project | Purpose |
| --- | --- |
| `InferenceEngine.Core` | Public API — interfaces, base class, options, model providers |
| `InferenceEngine.Examples` | Reference implementations (e.g. `FaceDetectionEngine`) |
| `InferenceEngine.Tests` | xUnit test suite with Coverlet coverage |

### Key Types

| Type | Location | Role |
| --- | --- | --- |
| `IInferenceEngine<TInput, TOutput>` | `InferenceEngine.Core/IInferenceEngine.cs` | Generic contract: `LoadAsync` + `PredictAsync` |
| `BaseInferenceEngine<TInput, TOutput>` | `InferenceEngine.Core/BaseInferenceEngine.cs` | Abstract base — ONNX session lifecycle, HW acceleration, warmup, disposal |
| `InferenceEngineOptions` | `InferenceEngine.Core/InferenceEngineOptions.cs` | Configuration record: GPU toggle, device ID, warmup, model provider |
| `IModelProvider` | `InferenceEngine.Core/ModelProviders/IModelProvider.cs` | Abstraction for model file acquisition |
| `LocalModelProvider` | `InferenceEngine.Core/ModelProviders/LocalModelProvider.cs` | Provides a model from a local file path |
| `DownloadingModelProvider` | `InferenceEngine.Core/ModelProviders/DownloadingModelProvider.cs` | Downloads from URLs with local caching and fallback |
| `ModelRegistry` | `InferenceEngine.Core/ModelProviders/ModelRegistry.cs` | JSON lookup (`models.json`) of model names → download URLs |
| `FaceDetectionEngine` | `InferenceEngine.Examples/FaceDetectionEngine.cs` | YOLO11 object detection with SkiaSharp preprocessing |

---

## Coding Conventions

### Language & Framework

- **C# 13 / .NET 10** — use latest language features (file-scoped namespaces are _not_ currently used; follow the existing style).
- **Nullable reference types** are enabled (`<Nullable>enable</Nullable>`).
- **Implicit usings** are enabled.
- XML doc comments are required on all public and protected members (`GenerateDocumentationFile` is on).

### Style Rules

- Use `PascalCase` for public/protected members, `_camelCase` for private fields.
- Prefer `var` when the type is obvious from the right-hand side.
- Keep methods short; extract complex logic into well-named private helpers.
- Use `readonly` on fields that don't change after construction.
- Dispose ONNX resources in `finally` blocks or `using` statements.
- Follow the existing pattern: `PreProcessWithContext` / `PostProcessWithContext` for engine pipelines.

### Error Handling

- Throw `ArgumentNullException` for null parameters with `nameof()`.
- Throw `InvalidOperationException` for invalid state (e.g. engine not loaded).
- Log warnings for non-critical failures (e.g. warmup failure, HW acceleration fallback).
- Use `ILogger` from `Microsoft.Extensions.Logging.Abstractions` — never `Console.WriteLine`.

### Testing

- Test framework: **xUnit** with `Xunit.SkippableFact` for conditional tests.
- Tests live in `InferenceEngine.Tests/` and reference both Core and Examples projects.
- Use `Skip.If(...)` for tests that require a GPU, network, or specific platform.
- Name tests: `MethodName_Scenario_ExpectedResult`.
- Run tests: `dotnet test` or the VS Code task `dotnet: build`.

---

## Build & CI

### Local Build

```bash
dotnet build                         # Build all projects
dotnet test                          # Run tests
dotnet test --collect:"XPlat Code Coverage"  # Tests + coverage
```

### CI/CD Workflows

| Workflow | File | Trigger | What It Does |
| --- | --- | --- | --- |
| Build & Test | `.github/workflows/build-test.yml` | Push / PR to `main`, `development` | Builds + tests on Ubuntu, Windows, macOS; uploads coverage |
| Publish NuGet | `.github/workflows/publish.yml` | Push to `release` | Packs `InferenceEngine.Core`, publishes to GitHub Packages + NuGet.org |

Both use a shared composite action at `.github/actions/build-and-test/action.yml`.

### Target Platforms

`win-x64`, `win-arm64`, `linux-x64`, `linux-arm64`, `osx-x64`, `osx-arm64`

---

## Repository Branch Strategy

### Branch Flow

```text
feature/<name> or personal branch (e.g., nam20485)
         │
    development
         │
       main
         │
     release
```

### Branch Naming

| Type | Pattern | Example |
| --- | --- | --- |
| Personal | `<username>` | `nam20485` |
| Feature | `feature/<descriptive-name>` | `feature/add-model-registry` |
| Sub-feature | `feature-<base>/<subject>` | `feature-model-registry/json-validation` |

### Pull Request Rules

- **PRs are required** for merges into `development`, `main`, and `release`.
- Feature / personal branches target `development`.
- CI must pass and all review comments must be resolved before merge.
- Automated agents may leave review comments — treat them as blocking.

### Merge Workflow

1. Branch from `development` (or use a personal branch for experimentation).
2. Commit, push, and open a PR targeting `development`.
3. Address review feedback; merge once approved.
4. `development` → `main` via PR when ready.
5. `main` → `release` for production deployment + NuGet publish.

---

## Agent Integration Patterns

This library is designed to be embedded inside AI agent systems. Below are canonical patterns.

### Vision Agent

```csharp
public class VisionAgent
{
    private readonly IInferenceEngine<byte[], List<Detection>> _detector;

    public async Task<AgentResponse> ProcessImageAsync(byte[] image)
    {
        var detections = await _detector.PredictAsync(image);
        return new AgentResponse { Objects = detections };
    }
}
```

### Multi-Modal Agent

```csharp
public class MultiModalAgent
{
    private readonly IInferenceEngine<byte[], List<Detection>> _vision;
    private readonly IInferenceEngine<string, string> _nlp;

    public async Task<string> AnalyzeSceneAsync(byte[] image, string query)
    {
        var objects = await _vision.PredictAsync(image);
        var context = $"Scene contains: {string.Join(", ", objects.Select(o => o.Label))}";
        return await _nlp.PredictAsync($"{context}\nQuery: {query}");
    }
}
```

### Custom Engine Checklist

1. Inherit from `BaseInferenceEngine<TInput, TOutput>`.
2. Override `ModelName` (and optionally `ModelDownloadUrls` or `ModelRegistry`).
3. Implement `PreProcessWithContext()` — convert domain input to ONNX tensors.
4. Implement `PostProcessWithContext()` — decode ONNX output to domain result.
5. Add a constructor that accepts `InferenceEngineOptions` and `ILogger<T>`.
6. Write unit tests in `InferenceEngine.Tests/`.

### Agent Lifecycle

| Phase | API | Notes |
| --- | --- | --- |
| Init | `engine.LoadAsync()` | Downloads model (if needed), creates ONNX session, runs warmup |
| Predict | `engine.PredictAsync(input)` | Thread-safe; auto-calls `LoadAsync` if not yet loaded |
| Cleanup | `engine.Dispose()` | Releases ONNX session and native memory |

### Performance Tips

- Set `WarmupOnLoad = true` in production to eliminate cold-start latency.
- Enable GPU acceleration for throughput-sensitive workloads.
- Always `Dispose()` engines when the agent shuts down.
- For high-throughput scenarios, consider batching inputs before calling `PredictAsync`.

---

## Common Tasks for Agents

### Adding a New Model to the Registry

Edit `InferenceEngine.Core/models.json`:

```json
{
  "name": "new-model.onnx",
  "description": "Description of the model",
  "source": "Source organization",
  "urls": [
    "https://primary-url/new-model.onnx",
    "https://fallback-url/new-model.onnx"
  ]
}
```

### Adding a New Engine

1. Create `InferenceEngine.Examples/NewEngine.cs` inheriting `BaseInferenceEngine<TInput, TOutput>`.
2. Create `InferenceEngine.Tests/NewEngineTests.cs` with unit tests.
3. If the model is publicly available, add it to `models.json`.

### Adding a NuGet Dependency

1. Add the `<PackageReference>` to the appropriate `.csproj`.
2. Prefer `Microsoft.Extensions.*` abstractions where applicable.
3. Use conditional `<ItemGroup>` for platform-specific packages (see `InferenceEngine.Core.csproj` for DirectML example).
