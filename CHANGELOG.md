# Changelog

All notable changes to the **InferenceEngine.Core** package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html) via the `{VERSION_PREFIX}[-suffix].{run_number}` scheme.

> **License transition:** package versions `1.1.x` and earlier were licensed **AGPL-3.0-or-later** and distributed from a private GitHub Packages feed. The **MIT License** applies from the first public `1.2.x` release onward (see [`LICENSE`](LICENSE) and [`THIRD-PARTY-NOTICES.txt`](THIRD-PARTY-NOTICES.txt)).

## [Unreleased]

Public-readiness phase: license, packaging, governance, documentation, and publishing changes preparing the first public `1.2.x` release.

### Added

- MIT-licensed package: `LICENSE` and `THIRD-PARTY-NOTICES.txt` ship inside the NuGet package
- Package metadata and icon: 512×512 package icon at the package root, corrected license expression, and package validation enabled; the Examples project is no longer packable
- Governance files: `SECURITY.md`, `CONTRIBUTING.md`, GitHub issue templates and pull-request template (this changelog is one of them)
- CI coverage reporting: HTML coverage report uploaded as a build artifact and a coverage summary posted as a pull-request comment, updated in place on re-runs (baseline 68.9% line / 54.3% branch)
- API documentation site: DocFX-built API docs plus the coverage site published to GitHub Pages

### Changed

- Dependency updates: `Microsoft.SourceLink.GitHub` 8.0.0 → 10.0.401 (fixes NU1902 / GHSA-23fw-v26w-5fgq), `Microsoft.ML.OnnxRuntime.DirectML` 1.23.0 → 1.24.1 (version-matched to the base ONNX Runtime package), `Microsoft.Extensions.Logging.Abstractions` 10.0.3 → 10.0.12, and `Microsoft.ML.Tokenizers` 1.0.3 → 2.0.0
- Publish workflow restructured: GitHub Releases (nupkg, snupkg, and `SHA256SUMS.txt`) plus NuGet.org publishing run behind the protected `release` environment; GitHub Packages publishing continues for `development`/`main` pre-releases

## [1.1.x] — private era (GitHub Packages, through 2026-09)

Private-era releases, distributed from a private GitHub Packages feed under **AGPL-3.0-or-later**. Final private stable: `1.1.29`. Superseded by the first public `1.2.x` release (MIT).

### Added

- `IInferenceEngine<TInput,TOutput>` engine contract: one testable `LoadAsync`/`PredictAsync` interface with full `IDisposable` lifecycle
- Hardware auto-selection with graceful CPU fallback: DirectML (Windows), CoreML (macOS), MIGraphX (Linux + AMD ROCm)
- Registry-driven model providers: model discovery, download, and local caching
- Warmup on load: dummy inference during `LoadAsync` to pre-compile GPU graphs
- Streaming autoregressive decode: `IAsyncEnumerable` output with KV-cache carry-back between steps
- Text inference engines: tokenizer-backed engines built on `Microsoft.ML.Tokenizers`
- Examples: YOLO face-detection demo engine and sample applications

[Unreleased]: https://github.com/intel-agency/inference-engine-lib/compare/development...HEAD
[1.1.x]: https://github.com/intel-agency/inference-engine-lib/pkgs/nuget/inferenceengine.core
