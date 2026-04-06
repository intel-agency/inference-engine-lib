# ROCm Integration — Implementation Plan for `inference-engine-lib`

> **Source repo:** `intel-agency/inference-engine-rocm` (this repo)
> **Target repo:** `intel-agency/inference-engine-lib`
> **Date:** 2026-04-06
> **Status:** inference-engine-rocm has a fully passing 4-job CI pipeline (build → pack → validate → publish). This document describes how to port those assets into inference-engine-lib.

> **⚠️ NOTE — Partially Outdated:** This plan was written before the integrated ROCm build assets were added to `inference-engine-lib`. The target repo now contains:
> - `scripts/build-rocm-native.sh` — the ROCm native build entry point (replaces the `compile_onnx_rocm_docker.sh` referenced below)
> - `.github/workflows/build-rocm-native.yml`, `validate-rocm-build.yml` — ROCm CI workflows (replace `build-rocm-linux.yml`)
> - `InferenceEngine.Core.IntegrationTests/` — Tier-1 validation test project
>
> Section 2 (Files to Port) should be treated as **historical context** for what was ported from `inference-engine-rocm`. The implementation checklist in Section 4 reflects the current status. Do not follow Section 2 instructions literally — the relevant files already exist under different names in the target repo.

---

## 1. Context & Background

### The Problem

There is **no official NuGet package** for the OnnxRuntime ROCm execution provider on Linux. Windows has `Microsoft.ML.OnnxRuntime.DirectML`, macOS gets CoreML in the base `Microsoft.ML.OnnxRuntime` — but Linux AMD GPU acceleration requires custom-compiled `.so` files:

| File | Purpose | Size |
|------|---------|------|
| `libonnxruntime.so` | Core OnnxRuntime shared library (ROCm-aware build) | ~23 MB |
| `libonnxruntime_providers_rocm.so` | ROCm execution provider plugin | ~93 MB |

### What inference-engine-rocm Solved

This repo contains:
1. A Dockerized compile script that builds ONNX Runtime v1.19.2 from source inside `rocm/dev-ubuntu-22.04:6.0.2`
2. A GitHub Actions workflow that runs the compile (~58 min), packs the `.so` into a NuGet package, validates the result, and publishes to GitHub Packages
3. Tier-1 integration tests that verify the `.so` files are valid ELF binaries with correct symbol exports

### What inference-engine-lib Needs

The target repo already has:
- `InferenceEngine.Core/` — The main library (identical source files)
- `InferenceEngine.Tests/` — Unit tests
- `InferenceEngine.Examples/` — FaceDetection example
- `.github/workflows/build-test.yml` — Cross-platform CI (Windows, macOS, Linux)
- `.github/workflows/publish.yml` — Publishes to GitHub Packages and NuGet.org
- `.github/actions/build-and-test/action.yml` — Composite action for build + test

**It does NOT have:**
- The ROCm compile script
- The ROCm build workflow
- Native `.so` bundling in the NuGet package
- The `runtimes/linux-x64/native/` structure in `.csproj`
- The Tier-1 validation test project

---

## 2. Files to Port

All source files are in `intel-agency/inference-engine-rocm/`. When this repo's folder is added to the inference-engine-lib workspace, they'll be directly accessible.

### 2.1 Compile Script (Copy As-Is)

**Source:** `inference-engine-rocm/scripts/compile_onnx_rocm_docker.sh`
**Target:** `inference-engine-lib/scripts/compile_onnx_rocm_docker.sh`

This is the battle-tested compile script. It survived 16 iterations of CI debugging. Copy it verbatim. Key design decisions baked in:

- **Docker image:** `rocm/dev-ubuntu-22.04:6.0.2` (pinned — ROCm 7.x breaks ORT v1.19.2 cmake)
- **ORT version:** `v1.19.2` (pinned git tag)
- **cmake:** `>=3.26,<4.0` (cmake 4.x breaks ORT's `cmake_minimum_required`)
- **Eigen:** Pre-fetched to exact commit `e7248b26a1ed53fa030c5c459f7ea095dfd276ac` (avoids GitLab tarball hash instability)
- **GPU architectures:** `gfx1030;gfx1031;gfx1100` (RDNA2 + RDNA3)
- **ComposableKernel:** Disabled (`onnxruntime_USE_COMPOSABLE_KERNEL=OFF` — causes header conflicts)
- **rocthrust:** Isolated install step (may not exist in all ROCm 6.x repos; if it fails, the rest of the build is unaffected)
- **`--build_shared_lib`** NOT `--build_wheel` (the wheel flag pulls in Python/numpy C headers we don't need)

### 2.2 GitHub Actions Workflow (Adapt)

**Source:** `inference-engine-rocm/.github/workflows/build-rocm-linux.yml`
**Target:** `inference-engine-lib/.github/workflows/build-rocm-linux.yml` (new file)

The workflow has 4 jobs in sequence:

```
build-rocm (57 min) → pack-nuget (30 sec) → validate-native (1 min) → publish-github-packages (15 sec)
```

**Adaptations for inference-engine-lib:**

1. **Trigger paths** — Add `InferenceEngine.Core/**` and `InferenceEngine.Core.IntegrationTests/**` to the `paths:` array (since the target repo has the C# source inline, not in a separate compile-only repo)
2. **SHA-pinned actions** — The rocm repo uses SHA-pinned actions. The target repo uses `@v4` tags. Either approach works; match the target repo's convention for consistency:
   - `actions/checkout@v4`
   - `actions/setup-dotnet@v4`
   - `actions/upload-artifact@v4`
   - `actions/download-artifact@v4`
3. **Pack versioning** — The rocm repo uses `1.0.0-rocm.${{ github.run_number }}`. In inference-engine-lib, align with the existing `${{ vars.VERSION_PREFIX }}.${{ github.run_number }}` pattern but add `-rocm` suffix: `${{ vars.VERSION_PREFIX }}.${{ github.run_number }}-rocm`
4. **Publish step** — The source workflow already includes a `publish-github-packages` job that pushes to `https://nuget.pkg.github.com/intel-agency/index.json` using `GITHUB_TOKEN`. It runs after validation passes and uses `--skip-duplicate` to avoid conflicts. Adapt the `permissions:` block to match inference-engine-lib's token configuration

### 2.3 Validation Test Project (Copy + Adapt)

**Source:** `inference-engine-rocm/InferenceEngine.Core.IntegrationTests/`
**Target:** `inference-engine-lib/InferenceEngine.Core.IntegrationTests/` (new directory)

Files to copy:
- `InferenceEngine.Core.IntegrationTests.csproj`
- `NativeLibraryValidationTests.cs`
- `Resources/identity.onnx` (65-byte minimal ONNX model)

**Adaptations:**
- The target repo uses `inference-engine-lib.slnx` (not `.sln`). Add the new project to the `.slnx`:
  ```xml
  <Project Path="InferenceEngine.Core.IntegrationTests/InferenceEngine.Core.IntegrationTests.csproj" />
  ```
- The existing `build-and-test` composite action filters out integration tests on Linux/macOS with `--filter "FullyQualifiedName!~FaceDetectionEngineTests"`. Extend the filter to also exclude the native validation tests (those should only run in the ROCm workflow, not on every PR):
  ```
  --filter "FullyQualifiedName!~FaceDetectionEngineTests&FullyQualifiedName!~NativeLibraryValidationTests"
  ```

### 2.4 `.csproj` Modification (Merge Carefully)

**Target file:** `inference-engine-lib/InferenceEngine.Core/InferenceEngine.Core.csproj`

The target repo's `.csproj` is nearly identical to the rocm repo's but is **missing the native asset inclusion**. Add this `ItemGroup` after the existing ones:

```xml
<!-- Include Native Assets for Linux ROCm -->
<ItemGroup>
  <None Include="runtimes/linux-x64/native/*.so" Pack="true" PackagePath="runtimes/linux-x64/native">
    <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
  </None>
</ItemGroup>
```

Also add `rocm` to the `<PackageTags>`:
```xml
<PackageTags>onnx;onnxruntime;ai;inference;machine-learning;deep-learning;directml;gpu;agents;rocm</PackageTags>
```

> **Important:** The `.so` files do NOT live in the git repo. They are injected at CI time by the `build-rocm-linux.yml` workflow's "Inject Native Assets" step. The `runtimes/linux-x64/native/` directory should be in `.gitignore`.

### 2.5 `.gitignore` Addition

Add to `inference-engine-lib/.gitignore`:
```
# ROCm native libs — injected by CI, not committed
InferenceEngine.Core/runtimes/
```

---

## 3. Workflow Architecture (Post-Integration)

After porting, inference-engine-lib will have these workflows:

| Workflow | Trigger | Jobs | Duration |
|----------|---------|------|----------|
| `build-test.yml` | push to main/dev, PRs | Build & Test (3 OS matrix) | ~5 min |
| `build-rocm-linux.yml` | push + path filter, manual | build-rocm → pack-nuget → validate-native → publish-github-packages | ~60 min |
| `publish.yml` | push to release, manual | Build → Pack → Publish (GitHub Packages + NuGet.org) | ~10 min |

The ROCm workflow runs independently and publishes the ROCm-specific NuGet package to GitHub Packages after validation passes. The standard `publish.yml` publishes the platform-agnostic package (CPU + DirectML + CoreML). Both packages coexist in the registry — the ROCm variant uses a `-rocm` version suffix (e.g., `1.0.0-rocm.42`).

---

## 4. Step-by-Step Implementation Checklist

### Phase 1: File Porting

- [x] ✅ Create `scripts/` directory in inference-engine-lib — implemented as `scripts/build-rocm-native.sh` (rewritten with helper functions, `--clean` flag, shallow clone)
- [x] ✅ Copy `scripts/compile_onnx_rocm_docker.sh` verbatim — adapted instead: modernized as `build-rocm-native.sh` targeting ORT v1.24.1
- [x] ✅ Copy `InferenceEngine.Core.IntegrationTests/` directory (3 files) — ported with `FindRepoRoot()` adapted for `.slnx`
- [x] ✅ Add `InferenceEngine.Core.IntegrationTests` project to `inference-engine-lib.slnx`

### Phase 2: Project Configuration

- [x] ✅ Add native `.so` inclusion `ItemGroup` to `InferenceEngine.Core/InferenceEngine.Core.csproj` — with `Condition="Exists(...)"` guard
- [x] ✅ Add `rocm` to `<PackageTags>` in `.csproj`
- [x] ✅ Add `InferenceEngine.Core/runtimes/` to `.gitignore`
- [x] ✅ Add `buildTransitive/InferenceEngine.Core.targets` — resolves native asset precedence over ORT CPU libs

### Phase 3: Workflow Creation

- [x] ✅ Create `.github/workflows/build-rocm-native.yml` — reusable `workflow_call` with L1 validation + artifact-level caching
- [x] ✅ Create `.github/workflows/validate-rocm-build.yml` — orchestrates build + verify + Tier-1 integration tests
- [x] ✅ Update `.github/workflows/publish.yml` — integrates ROCm build, adds `validate-native` gate before publish
- [x] ✅ Update `.github/actions/build-and-test/action.yml` test filter to exclude `NativeLibraryValidationTests`

### Phase 4: Verify

- [ ] Commit and push to trigger workflows
- [ ] Confirm `build-rocm-native` job succeeds (~58 min first run, seconds on cache hit)
- [ ] Confirm all L1 inline validations pass (existence, size, ELF headers)
- [ ] Confirm `validate-native` job succeeds (Tier-1 integration tests pass)
- [ ] Confirm `build-test.yml` still passes on all 3 OS platforms (no regression from the new test project)

### Phase 5: Publish

- [ ] Confirm `publish-github-packages` job succeeds (gated on `validate-native`)
- [ ] Confirm `publish-nuget-org` job succeeds (gated on `validate-native`)
- [ ] Confirm package appears at `https://github.com/orgs/intel-agency/packages?repo_name=inference-engine-lib`

---

## 5. Known Constraints & Gotchas

### Build Time
The ROCm compile job takes **~58 minutes** on `ubuntu-latest`. This is unavoidable — it's compiling ONNX Runtime from C++ source with the HIP compiler. The workflow uses `workflow_dispatch` so it can be triggered manually rather than on every push.

### Docker Image Pinning
**Do not** update `rocm/dev-ubuntu-22.04:6.0.2` without testing. ROCm 7.x images break ORT v1.19.2's cmake configuration. If you need to update ORT version, the Docker image version will need to be re-validated.

### cmake Version
Must be `>=3.26,<4.0`. cmake 4.x changed `cmake_minimum_required()` behavior, breaking every ORT dependency that uses an older minimum version.

### Disk Space
The GitHub Actions runner needs ~40 GB free. The workflow's first step frees space by removing `/usr/share/dotnet`, `/opt/ghc`, and Docker images.

### No GPU in CI
Standard GitHub Actions runners have no AMD GPU. The validation tests are designed for this:
- ELF format checks, symbol export checks, and ORT session loading all work without a GPU
- The ROCm EP load test expects a clean `OnnxRuntimeException` (not a SIGABRT/native crash) — proving the `.so` loads and registers itself correctly even without hardware

### The 9 Tier-1 Validation Tests

| # | Test | What it proves |
|---|------|---------------|
| 1 | `LibOnnxRuntime_Exists` | `.so` file was built and uploaded |
| 2 | `LibOnnxRuntimeProvidersRocm_Exists` | ROCm provider `.so` was built |
| 3 | `BothLibs_AreNonEmpty` | Files are > 1 MB (not stubs/corrupt) |
| 4 | `LibOnnxRuntime_IsElf64` | Valid ELF 64-bit shared object |
| 5 | `LibOnnxRuntimeProvidersRocm_IsElf64` | Valid ELF 64-bit shared object |
| 6 | `LibOnnxRuntime_ExportsOrtGetApiBase` | Core ORT API symbol is exported |
| 7 | `LibOnnxRuntimeProvidersRocm_ExportsRocmProvider` | ROCm EP registration symbols present |
| 8 | `InferenceSession_LoadsIdentityModel_WithoutCrash` | ORT managed layer + native lib work together |
| 9 | `InferenceSession_RunsIdentityModel_ProducesCorrectOutput` | End-to-end inference produces `[42.0]` from identity model |

Tests 1-7 require `NATIVE_LIBS_DIR` env var pointing at the downloaded artifact directory. Tests 8-9 use the NuGet-bundled ORT native libs (CPU fallback).

An additional test (`InferenceSession_WithRocmEP_ThrowsCleanExceptionNotCrash`) attempts to load the ROCm EP and asserts it either succeeds (if GPU present) or throws a managed `OnnxRuntimeException` (no SIGABRT).

---

## 6. Reference: Source Files in This Repo

When inference-engine-rocm is added to the workspace, these files are available at:

| File | Description |
|------|-------------|
| `inference-engine-rocm/scripts/compile_onnx_rocm_docker.sh` | The 150-line battle-tested compile script |
| `inference-engine-rocm/.github/workflows/build-rocm-linux.yml` | 4-job workflow (build → pack → validate → publish) |
| `inference-engine-rocm/InferenceEngine.Core.IntegrationTests/NativeLibraryValidationTests.cs` | 9 Tier-1 validation tests |
| `inference-engine-rocm/InferenceEngine.Core.IntegrationTests/InferenceEngine.Core.IntegrationTests.csproj` | Test project file |
| `inference-engine-rocm/InferenceEngine.Core.IntegrationTests/Resources/identity.onnx` | 65-byte minimal ONNX model |
| `inference-engine-rocm/InferenceEngine.Core/InferenceEngine.Core.csproj` | Reference `.csproj` with native asset inclusion |

### Successful CI Run Reference
- **Run:** [24022362384](https://github.com/intel-agency/inference-engine-rocm/actions/runs/24022362384)
- **Build ROCm:** 57m43s, `success`
- **Pack NuGet:** 31s, `success`
- **Artifacts:** `native-rocm-libs` (116 MB), `nuget-package-rocm`
