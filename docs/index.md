# InferenceEngine.Core

**A cross-platform .NET 10 library for AI inference with ONNX Runtime** — hardware acceleration,
automatic fallback, and built-in diagnostics for provider selection.

`InferenceEngine.Core` wraps ONNX Runtime behind three small API surfaces (one-shot, streaming,
and out-of-process), probes the host for the best execution provider (MIGraphX on AMD ROCm, 
DirectML on Windows, CoreML on macOS, CPU everywhere), and degrades gracefully when a provider's
native stack is absent.

- **Source:** [intel-agency/inference-engine-lib on GitHub](https://github.com/intel-agency/inference-engine-lib)
- **Docs site:** <https://intel-agency.github.io/inference-engine-lib/>

## Install

```bash
dotnet add package InferenceEngine.Core
```

Prerelease (`-dev` / `-rc`) builds are published to GitHub Packages — see the
[README](https://github.com/intel-agency/inference-engine-lib#readme) for the feed setup.

## Guides

- [**Consumer integration guide**](consumer-integration-guide.md) — start here: which of the
  three API surfaces fits your app, and how to wire each one up.
- [**Streaming inference guide**](streaming-inference-guide.md) — chunked/token-by-token results
  for LLM/SLM runners, speech engines, and step-wise diffusion.
- [**Text engines guide**](text-engines-guide.md) — string-in/string-out embedding and generation
  engines with tokenization owned by the library.

## API reference

Full generated API documentation: **[API reference](../api/toc.yml)**.

## Coverage

Live line/branch coverage from `main`, rendered by ReportGenerator:
**[coverage report](https://intel-agency.github.io/inference-engine-lib/coverage/)**.
