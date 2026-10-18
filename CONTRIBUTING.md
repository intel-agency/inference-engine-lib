# Contributing to InferenceEngine.Core

Thanks for your interest in contributing. This document covers how to build and test the library, the project's branching and merge policy, and what we expect in a contribution.

**Please open an issue first** before starting significant work (new engines or features, public-API changes, CI changes, packaging or versioning changes) so we can agree on the approach. Small documentation fixes don't need an issue.

## Prerequisites

- The **.NET 10 SDK**

## Restoring dependencies (current state)

> **Note — private feed required while this repository is pre-public.** Restores that include the Linux x64 runtime need the private GitHub Packages feed configured in [`nuget.config`](nuget.config) (the feed backs `InferenceEngine.ROCm.Runtime.linux-x64`, referenced on the `linux-x64` RID). Set the `NUGET_AUTH_TOKEN` environment variable to a personal access token with the `read:packages` scope before such restores; Windows/macOS restores resolve entirely from nuget.org. This requirement disappears once the ROCm runtime dependency is published on nuget.org — the public-readiness work makes restore fully public and tokenless.

## Building and testing

```bash
# Build
dotnet build

# Standard test suite — excludes the network/GPU integration tests
# (FaceDetectionEngineTests) and the native-library validation tests
# (NativeLibraryValidationTests, requires the ROCm-built libonnxruntime.so)
dotnet test --filter "FullyQualifiedName!~FaceDetectionEngineTests&FullyQualifiedName!~NativeLibraryValidationTests"

# Full suite with coverage, as CI runs it
dotnet test --collect:"XPlat Code Coverage"
```

## Validation tiers

- **Standard suite** — unit and engine-contract tests; runs in CI on every push/PR across `ubuntu`, `windows`, and `macos`.
- **GPU-dependent paths** — tests that require actual GPU inference (DirectML, CoreML, MIGraphX) are skip-gated on hardware presence and **auto-skip on GitHub-hosted runners**; hardware-independent provider-selection and CPU-fallback logic runs everywhere.

To exercise MIGraphX inference end-to-end, the repository defines a self-hosted Linux x64 runner profile with an AMD GPU — registered with labels `self-hosted, linux, x64, amd-gpu` and gated behind the `ROCM_RUNNER_ENABLED` repository variable. The runner needs AMD ROCm userspace (`libmigraphx_c.so.3` + `libamdhip64.so.7`), `/dev/kfd` accessible, and a `NUGET_AUTH_TOKEN` for the private-feed restore. See [`.github/workflows/rocm-gpu-test.yml`](.github/workflows/rocm-gpu-test.yml).

## CI overview

| Workflow | Trigger | Description |
| :--- | :--- | :--- |
| **Build & Test** | Push / PR to `development`, `main`, `release` (+ manual) | Builds and tests on `ubuntu`, `windows`, `macos`; collects coverage and posts a coverage summary to pull requests |
| **ROCm GPU Inference Test** | Push to `development`, `main`, `release`, or manual | Runs GPU inference on the self-hosted AMD runner (skipped until the runner is enabled) |
| **Publish** | Push to `development`, `main`, `release` (paths-filtered, + manual) | Packs and publishes `InferenceEngine.Core` — GitHub Packages on dev/rc/stable pushes; NuGet.org publishing and the GitHub Release run behind the protected `release` environment |

## Branching and merge policy

- Branch from **`development`**: `feature/<short-topic>` or `dev/<short-topic>`.
- Integration between long-lived branches (`development` → `main` → `release`) happens via **merge commits only**. Squash merges and rebases are **banned** — they rewrite history and destroy the audit trail.
- **Every commit must be GPG-signed** (`git config commit.gpgsign true`). Unsigned commits are rejected.
- CI must be green and review complete before merge. **Automated review comments (bot/AI reviewer findings) are blocking** — resolve them like human review feedback.

## Pull requests

Open pull requests against `development` using the [pull request template](.github/PULL_REQUEST_TEMPLATE.md). It asks for a summary, the package impact (metadata, dependencies, versioning), and the validation you performed.

## Reporting vulnerabilities

Do not open public issues for security problems — see [SECURITY.md](SECURITY.md) for private vulnerability reporting.
