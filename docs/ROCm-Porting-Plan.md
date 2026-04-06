# ROCm Linux Shared Library Build — Porting Plan

> **Source repo:** `intel-agency/inference-engine-rocm`
> **Target repo:** `intel-agency/inference-engine-lib`
> **Branch:** `mn/linux-so-ci`
> **Date:** 2026-04-06

---

## 1. Current State Assessment

### Already Ported to inference-engine-lib

| Asset | Status | Notes |
|-------|--------|-------|
| `scripts/build-rocm-native.sh` | **Done** | Rewritten from `compile_onnx_rocm_docker.sh` — modernized with helper functions, `--clean` flag, improved logging, and shallow clone (`--depth 1`) |
| `.github/workflows/build-rocm-native.yml` | **Done** | Reusable `workflow_call` variant with L1 validation (existence, size, ELF header checks) |
| `.github/workflows/validate-rocm-build.yml` | **Done** | Trigger workflow on push to `development`/`main`; calls `build-rocm-native.yml` |
| `.github/workflows/publish.yml` | **Done** | Integrates ROCm build (`build-rocm-native`) into the pack → publish pipeline; injects `.so` into NuGet |
| `InferenceEngine.Core.csproj` — native asset inclusion | **Done** | `runtimes/linux-x64/native/*.so` with `Condition="Exists(...)"` guard |
| `InferenceEngine.Core.csproj` — `rocm` tag | **Done** | Added `rocm` to `<PackageTags>` |
| `buildTransitive/InferenceEngine.Core.targets` | **Done** | Resolves native asset precedence over `Microsoft.ML.OnnxRuntime` CPU libs |
| `.gitignore` — `InferenceEngine.Core/runtimes/` | **Done** | Prevents committing CI-injected `.so` files |
| `.gitignore` — `build/` | **Done** | Covers local `build-rocm-native.sh` output |
| `InferenceEngine.Tests/RocmProviderTests.cs` | **Done** | L2 load-time validation (ROCm provider append, CPU fallback, session options) |
| `build-and-test/action.yml` — NuGet caching | **Done** | Uses `actions/cache@v4` with `hashFiles('**/*.csproj')` key |
| `InferenceEngine.Core.IntegrationTests/` project | **Done** | Ported with adapted `FindRepoRoot()` for `.slnx` |
| `inference-engine-lib.slnx` entry | **Done** | IntegrationTests project added to solution |
| `build-and-test/action.yml` — `NativeLibraryValidationTests` filter | **Done** | Excluded from Linux/macOS default test runs |
| `validate-rocm-build.yml` — `validate-native` job | **Done** | Runs Tier-1 tests after ROCm build |
| `publish.yml` — `validate-native` gate | **Done** | Gates both `publish-github-packages` and `publish-nuget-org` |
| `build-rocm-native.yml` — artifact caching | **Done** | `actions/cache@v4` keyed on `hashFiles('scripts/build-rocm-native.sh')` — skips 60-min build on hit |

### Not Yet Ported

All items from this section have been implemented. See "Already Ported" table above.

### Remaining (Optional Hardening)

| Asset | Priority | Notes |
|-------|----------|-------|
| Pin Docker image version in `build-rocm-native.sh` | Optional | `latest` risks breaking changes; consider pinning `6.0.2` |
| Add Eigen pre-fetch to compile script | Optional | Avoids GitLab tarball hash instability |
| Add `onnxruntime_USE_COMPOSABLE_KERNEL=OFF` | Optional | Prevents header conflicts found in source repo |
| Add rocthrust isolation install step | Optional | Fallback for ROCm version compatibility |
| Pin cmake range `>=3.26,<4.0` | Optional | Prevents cmake 4.x incompatibility with ORT |

### Key Differences Between Source and Target

| Aspect | inference-engine-rocm | inference-engine-lib |
|--------|----------------------|---------------------|
| Build script | `compile_onnx_rocm_docker.sh` (inline, single-file) | `build-rocm-native.sh` (structured, helper functions, `--clean` support) |
| ORT version | `v1.19.2` | `v1.24.1` |
| Docker image | `rocm/dev-ubuntu-22.04:6.0.2` (pinned) | `rocm/dev-ubuntu-22.04:latest` |
| GPU architectures | `gfx1030;gfx1031;gfx1100` | `gfx1030;gfx1031;gfx1032;gfx1100;gfx1101` |
| Eigen pre-fetch | Manual git clone to avoid hash instability | Not present (relies on ORT build.sh) |
| ComposableKernel | Explicitly `OFF` | Not specified |
| rocthrust isolation | Explicit fallback install | Not present |
| Workflow structure | 4-job monolith | Reusable `workflow_call` + orchestration workflows |
| Action pins | SHA-pinned (`@SHA`) | Tag-pinned (`@v4`) |
| Pack versioning | `1.0.0-rocm.${{ run_number }}` | `${{ vars.VERSION_PREFIX }}.${{ run_number }}` (no `-rocm` suffix) |
| Solution file | `.sln` | `.slnx` |
| Test filter | No exclusion (dedicated project) | Excludes `FaceDetectionEngineTests` on Linux/macOS |
| L1 validation | In workflow inline (`validate-native` job) + integration test project | In workflow inline (existence, size, ELF steps) |
| L2 validation | `NativeLibraryValidationTests.cs` (symbol exports, ORT load, inference) | `RocmProviderTests.cs` (provider config, CPU fallback) |

---

## 2. Remaining Porting Tasks

### Phase 1: Integration Tests Project

1. **Copy `InferenceEngine.Core.IntegrationTests/`** from `inference-engine-rocm/`
   - `InferenceEngine.Core.IntegrationTests.csproj`
   - `NativeLibraryValidationTests.cs`
   - `Resources/identity.onnx`

2. **Adapt for inference-engine-lib:**
   - Change `FindRepoRoot()` to search for `inference-engine-lib.slnx` instead of `inference-engine-rocm.sln`
   - Verify project reference path to `InferenceEngine.Core.csproj` resolves correctly

3. **Add to solution:**
   ```xml
   <!-- inference-engine-lib.slnx -->
   <Project Path="InferenceEngine.Core.IntegrationTests/InferenceEngine.Core.IntegrationTests.csproj" />
   ```

4. **Exclude from default test runs** in `build-and-test/action.yml`:
   ```bash
   # Linux/macOS filter
   --filter "FullyQualifiedName!~FaceDetectionEngineTests&FullyQualifiedName!~NativeLibraryValidationTests"
   ```

### Phase 2: Minor csproj Update

5. **Add `rocm` to `<PackageTags>`** in `InferenceEngine.Core.csproj`:
   ```xml
   <PackageTags>onnx;onnxruntime;ai;inference;machine-learning;deep-learning;directml;gpu;agents;rocm</PackageTags>
   ```

### Phase 3: Build Script Hardening (Optional)

Review whether `build-rocm-native.sh` should incorporate these battle-tested patterns from `compile_onnx_rocm_docker.sh`:

6. **Pin Docker image version** — `rocm/dev-ubuntu-22.04:latest` risks breaking changes; consider pinning `6.0.2` or a tested version
7. **Add Eigen pre-fetch** — avoids GitLab tarball hash instability that caused failures in the source repo
8. **Add `onnxruntime_USE_COMPOSABLE_KERNEL=OFF`** — prevents header conflicts discovered during 16 iterations of CI debugging
9. **Add rocthrust isolation** — separate install step with fallback for ROCm version compatibility
10. **Pin cmake range** — `>=3.26,<4.0` to prevent cmake 4.x incompatibility with ORT

---

## 3. Validation Strategy

### Validation Tiers

The ROCm build produces native binaries that cannot be tested on typical CI runners (no AMD GPU). Validation is layered:

#### Tier 1 — Binary Integrity (CI, no GPU required)

| Check | Location | Description |
|-------|----------|-------------|
| Existence | `build-rocm-native.yml` inline step | Confirm `libonnxruntime.so` and `libonnxruntime_providers_rocm.so` exist |
| Minimum size | `build-rocm-native.yml` inline step | `libonnxruntime.so` ≥ 10 MB, `libonnxruntime_providers_rocm.so` ≥ 50 MB |
| ELF headers | `build-rocm-native.yml` inline step | `file` command confirms `ELF 64-bit LSB shared object` |
| Artifact placement | `publish.yml` `Verify ROCm Artifacts Placement` step | Confirms `.so` files land in `runtimes/linux-x64/native/` before pack |

These checks already exist in the ported `build-rocm-native.yml` workflow.

#### Tier 1+ — Native Integration Tests (CI, no GPU required)

| Check | Source | Description |
|-------|--------|-------------|
| File existence + size > 1MB | `NativeLibraryValidationTests.cs` | Programmatic check both `.so` files exist and are substantial |
| ELF 64-bit shared object | `NativeLibraryValidationTests.cs` | Uses `file` command to verify binary format |
| Symbol exports (`OrtGetApiBase`) | `NativeLibraryValidationTests.cs` | `nm -D` verifies core ORT API symbol is exported |
| Symbol exports (ROCm EP) | `NativeLibraryValidationTests.cs` | `nm -D` verifies ROCm execution provider symbols |
| ORT environment init | `NativeLibraryValidationTests.cs` | `OrtEnv.Instance()` loads without crash |
| Identity model load | `NativeLibraryValidationTests.cs` | CPU session loads the 65-byte test model |
| Identity model inference | `NativeLibraryValidationTests.cs` | Input `42.0f` → output `42.0f` roundtrip |
| ROCm EP graceful failure | `NativeLibraryValidationTests.cs` | Requesting ROCm EP without GPU throws `OnnxRuntimeException`, not SIGABRT |

**Status:** Ported. `NativeLibraryValidationTests.cs` is now in `InferenceEngine.Core.IntegrationTests/` and wired into both `validate-rocm-build.yml` and `publish.yml` as a gate before publishing.

#### Tier 2 — Provider Configuration (CI, no GPU required)

| Check | Source | Description |
|-------|--------|-------------|
| ROCm provider append | `RocmProviderTests.cs` | Engine configures ROCm EP without crash on Linux x64 |
| CPU fallback | `RocmProviderTests.cs` | Engine loads successfully when ROCm is unavailable |
| Session options structure | `RocmProviderTests.cs` | Runtime-configured session options are valid |

**Status:** Already ported and running in `build-test.yml`.

#### Tier 3 — Full GPU Execution (Self-hosted runner, requires AMD GPU hardware)

| Check | Status | Description |
|-------|--------|-------------|
| ROCm EP session creation | **Not implemented** | Create inference session with ROCm EP on real GPU |
| Model inference on GPU | **Not implemented** | Run FaceDetection model on AMD GPU, verify results |
| Multi-model throughput | **Not implemented** | Load/run multiple models to validate memory management |

Tier 3 requires self-hosted runners with AMD GPU hardware and is out of scope for this porting plan.

### Validation Workflow Integration

After porting Phase 1, the workflows will validate as follows:

```
build-test.yml (PR / push to main/dev)
└─ Runs: RocmProviderTests (L2, via existing test project)
└─ Excludes: NativeLibraryValidationTests, FaceDetectionEngineTests

validate-rocm-build.yml (push to development/main)
└─ build-rocm-native.yml (workflow_call)
   ├─ Inline L1: existence, size, ELF checks
   └─ Upload artifacts
└─ verify-rocm-artifacts (downloads + logs)
└─ validate-native job: runs NativeLibraryValidationTests with NATIVE_LIBS_DIR ✅

publish.yml (push to release)
├─ build-and-test (3 OS matrix, excludes integration tests)
├─ build-rocm-native (workflow_call, with artifact cache)
├─ pack (injects .so, dotnet pack)
├─ validate-native (runs NativeLibraryValidationTests before publish) ✅
├─ publish-github-packages (gated on validate-native)
└─ publish-nuget-org (gated on validate-native)
```

The key addition is running `NativeLibraryValidationTests` **after** the ROCm build and **before** publish, creating a gate that prevents publishing a package with broken native libraries.

---

## 4. Caching Strategy

The ROCm Docker build is the dominant cost: **45–60 minutes** per run. The build involves:
1. Pulling the Docker image (~5 GB)
2. Installing apt packages inside the container (~2 min)
3. Cloning ONNX Runtime + submodules (~3 min)
4. Compiling C++/HIP code (~40–55 min)

### 4.1 Docker Image Layer Cache

**Impact: Save ~5 min on image pull**

GitHub Actions runners don't persist Docker images between runs. Use Docker layer caching:

```yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v3

- name: Cache Docker layers
  uses: actions/cache@v4
  with:
    path: /tmp/.docker-cache
    key: ${{ runner.os }}-docker-rocm-${{ hashFiles('scripts/build-rocm-native.sh') }}
    restore-keys: |
      ${{ runner.os }}-docker-rocm-
```

However, since we're using `docker run` (not `docker build`) with a prebuilt base image, this has limited value. A more effective approach is to **pre-pull and cache** the image:

```yaml
- name: Cache ROCm Docker image
  id: cache-docker
  uses: actions/cache@v4
  with:
    path: /tmp/rocm-docker-image.tar
    key: rocm-docker-${{ env.ROCM_DOCKER_IMAGE }}

- name: Pull and save Docker image
  if: steps.cache-docker.outputs.cache-hit != 'true'
  run: |
    docker pull ${{ env.ROCM_DOCKER_IMAGE }}
    docker save ${{ env.ROCM_DOCKER_IMAGE }} -o /tmp/rocm-docker-image.tar

- name: Load cached Docker image
  if: steps.cache-docker.outputs.cache-hit == 'true'
  run: docker load -i /tmp/rocm-docker-image.tar
```

> **Caveat:** The Docker image is ~5 GB; `actions/cache` has a 10 GB total limit per repo. This would consume half of it. Weigh against other cache needs. With `latest` tag, the cache also invalidates every time the upstream image is updated — consider pinning a tag (e.g., `6.0.2`) for stable caching.

### 4.2 ONNX Runtime Source Cache

**Impact: Save ~3 min on clone + submodule init**

The current script clones ONNX Runtime on every run. Cache the checkout:

```yaml
- name: Cache ONNX Runtime source
  uses: actions/cache@v4
  with:
    path: build/onnxruntime-rocm-build/onnxruntime
    key: ort-source-${{ env.ONNX_RUNTIME_VERSION }}
```

The `build-rocm-native.sh` already supports reuse: it checks for an existing clone and syncs to the correct tag. This cache would skip the initial `git clone --depth 1 --recursive` (which still fetches ~1.5 GB of submodules).

> **Caveat:** With `--depth 1`, the clone is smaller but submodules can still be large. Size the cache expectation at ~2–3 GB.

### 4.3 ccache / sccache for C++ Compilation

**Impact: Save 30–50 min on rebuild (when source hasn't changed)**

This is the **highest-value cache** and addresses the dominant cost. `ccache` or `sccache` caches compiled object files keyed on preprocessed source + compiler flags. On a cache hit, recompilation drops from ~50 min to ~5 min.

#### Implementation approach:

1. **Install ccache inside the Docker container** (add to the generated compile script):
   ```bash
   apt-get install -y ccache
   export PATH="/usr/lib/ccache:$PATH"
   export CCACHE_DIR=/code/ccache
   export CCACHE_MAXSIZE=5G
   ```

2. **Mount ccache directory from host** (persisted via `actions/cache`):
   ```yaml
   - name: Cache ccache artifacts
     uses: actions/cache@v4
     with:
       path: build/onnxruntime-rocm-build/ccache
       key: ccache-rocm-${{ env.ONNX_RUNTIME_VERSION }}-${{ hashFiles('scripts/build-rocm-native.sh') }}
       restore-keys: |
         ccache-rocm-${{ env.ONNX_RUNTIME_VERSION }}-
         ccache-rocm-
   ```

3. **Pass ccache dir to Docker** (already mounted via `-v`):
   ```bash
   docker run --rm \
     -v "${work_dir_path}:/code" \
     -e CCACHE_DIR=/code/ccache \
     ...
   ```

4. **Modify compile script** to configure ccache as the compiler launcher:
   ```bash
   --cmake_extra_defines CMAKE_C_COMPILER_LAUNCHER=ccache \
   --cmake_extra_defines CMAKE_CXX_COMPILER_LAUNCHER=ccache \
   --cmake_extra_defines CMAKE_HIP_COMPILER_LAUNCHER=ccache \
   ```

#### Expected cache behavior:

| Scenario | Build Time | Cache Hit Rate |
|----------|-----------|----------------|
| First run (cold cache) | ~55 min | 0% |
| Re-run, same ORT version + script | ~5–8 min | ~95% |
| ORT version bump | ~45 min | ~20% (shared infrastructure objects) |
| Script change (cmake flags) | ~30 min | ~50% (unchanged compilation units) |

> **Caveat:** `actions/cache` has a 10 GB limit per repo. ccache artifacts for a ROCm build can reach 3–5 GB. Combined with Docker image cache and source cache, this could exceed the limit. Prioritize ccache over other caches — it provides the largest time savings.

### 4.4 CMake Build Directory Cache

**Impact: Save ~5 min on cmake configure step**

Cache the CMake build directory to skip reconfiguration:

```yaml
- name: Cache CMake build directory
  uses: actions/cache@v4
  with:
    path: build/onnxruntime-rocm-build/onnxruntime/build/Linux/Release
    key: cmake-rocm-${{ env.ONNX_RUNTIME_VERSION }}-${{ hashFiles('scripts/build-rocm-native.sh') }}
    restore-keys: |
      cmake-rocm-${{ env.ONNX_RUNTIME_VERSION }}-
```

> **Caveat:** CMake build directories are fragile — stale caches can cause cryptic build failures. Only viable if the ORT version and cmake flags are part of the cache key. Consider this **lower priority** than ccache.

### 4.5 Artifact Reuse Across Workflows

**Impact: Eliminate redundant 60-min builds**

Currently, both `validate-rocm-build.yml` and `publish.yml` trigger independent ROCm builds. If the `.so` files haven't changed (same ORT version, same script, same Docker image), the build is redundant.

#### Option A: Conditional build skip

Use `actions/cache` to store the final `.so` artifacts keyed on inputs:

```yaml
- name: Check for cached ROCm artifacts
  id: rocm-cache
  uses: actions/cache@v4
  with:
    path: build/linux-x64/native/*.so
    key: rocm-artifacts-${{ hashFiles('scripts/build-rocm-native.sh') }}-${{ env.ONNX_RUNTIME_VERSION }}

- name: Build ROCm native libraries
  if: steps.rocm-cache.outputs.cache-hit != 'true'
  run: ./scripts/build-rocm-native.sh
```

This completely skips the build when the script and ORT version haven't changed.

#### Option B: Workflow-level artifact promotion

Store validated `.so` artifacts as release assets or in a package registry. `publish.yml` downloads pre-validated artifacts instead of rebuilding:

```yaml
# In publish.yml, replace build-rocm-native call with:
- name: Download validated ROCm artifacts
  uses: actions/download-artifact@v4
  # Or: gh release download --pattern '*.so' ...
```

This requires a workflow to produce and tag the artifacts separately.

### 4.6 Caching Priority Matrix (Updated)

| Cache | Time Saved | Cache Size | Complexity | Status |
|-------|-----------|------------|------------|--------|
| **Artifact reuse (skip rebuild)** | 45–60 min | ~120 MB | Low | **Implemented** in `build-rocm-native.yml` |
| ~~ccache (C++ objects)~~ | ~~30–50 min~~ | ~~3–5 GB~~ | ~~Medium~~ | **Dropped** — CI uses clean builds; artifact cache is strictly better |
| **ORT source cache** | 3 min | 2–3 GB | Low | Optional (P1) |
| **Docker image cache** | 5 min | ~5 GB | Low | Optional (P2, budget concern) |
| ~~CMake build dir cache~~ | ~~5 min~~ | ~~1–2 GB~~ | ~~Medium~~ | **Dropped** — fragile, artifact cache covers this |

### 4.7 Cache Budget Constraint (Updated)

GitHub Actions provides **10 GB total cache** per repository. Current allocation:

| Cache | Allocated |
|-------|-----------|
| NuGet packages (existing) | ~0.5 GB |
| ROCm artifact final output | ~0.2 GB |
| **Total** | **~0.7 GB** |

This leaves ~9.3 GB headroom for future caching needs (ORT source, Docker image).

---

## 5. Implementation Checklist

### Phase 1: Port Integration Tests ✅ Complete

- [x] Copy `InferenceEngine.Core.IntegrationTests/` directory from `inference-engine-rocm/`
- [x] Adapt `FindRepoRoot()` in `NativeLibraryValidationTests.cs` to find `inference-engine-lib.slnx`
- [x] Add project to `inference-engine-lib.slnx`
- [x] Update `build-and-test/action.yml` to exclude `NativeLibraryValidationTests` on Linux/macOS
- [x] Add `validate-native` job to `validate-rocm-build.yml` (runs integration tests after build)
- [x] Add `validate-native` job to `publish.yml` between `pack` and `publish-*` jobs
- [x] Add `rocm` to `<PackageTags>` in `InferenceEngine.Core.csproj`

### Phase 2: Caching ✅ Complete

- [x] Add artifact-level cache to `build-rocm-native.yml` (skip 60-min build when script unchanged)
- [x] ~~ccache~~ — Dropped: CI runs are always clean builds on ephemeral runners; artifact-level cache is strictly better
- [ ] ORT source tree cache (P1, optional — saves ~3 min)
- [ ] Docker image cache (P2, optional — budget concern)

### Phase 3: Build Script Hardening (Optional)

- [ ] Pin Docker image version in `build-rocm-native.sh` (e.g., `6.0.2`)
- [ ] Add Eigen pre-fetch to generated compile script
- [ ] Add `onnxruntime_USE_COMPOSABLE_KERNEL=OFF` cmake define
- [ ] Add rocthrust isolation install step
- [ ] Add cmake version constraint (`>=3.26,<4.0`)

### Phase 4: Verification

- [ ] Push changes to trigger `validate-rocm-build.yml`
- [ ] Confirm `build-rocm-native` job succeeds (~58 min first run, seconds on cache hit)
- [ ] Confirm all L1 inline validations pass (existence, size, ELF headers)
- [ ] Confirm `NativeLibraryValidationTests` pass
- [ ] Confirm `build-test.yml` still passes on all 3 OS platforms (no regression)
- [ ] Confirm `RocmProviderTests` continue passing on Linux
- [ ] Test `publish.yml` end-to-end on `release` branch
- [ ] Confirm package appears at `https://github.com/orgs/intel-agency/packages?repo_name=inference-engine-lib`

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ORT `v1.24.1` build breaks with ROCm | Medium | High | Test locally first; fall back to `v1.19.2` if needed |
| `rocm/dev-ubuntu-22.04:latest` image update breaks build | Medium | High | Pin to tested version (e.g., `6.0.2`) |
| ~~ccache exceeds `actions/cache` 10 GB limit~~ | — | — | Dropped: artifact-level cache used instead |
| ~~CMake build dir cache causes stale-config errors~~ | — | — | Dropped: artifact-level cache used instead |
| IntegrationTests project `RuntimeIdentifier=linux-x64` breaks Windows/macOS `dotnet build` | Medium | Medium | Exclude via `--filter` or condition in solution; test cross-platform build |
| `buildTransitive` targets conflict with consumer projects | Low | High | Test in a clean consumer project; document known interactions |

---

## 7. References

- [ROCm Integration Implementation Plan](ROCm-Integration-Implementation-Plan.md) — Original porting specification
- [inference-engine-rocm ARCHITECTURE.md](../../inference-engine-rocm/ARCHITECTURE.md) — Source repo architecture
- [inference-engine-rocm build-rocm-linux.yml](../../inference-engine-rocm/.github/workflows/build-rocm-linux.yml) — Source CI workflow
- [inference-engine-rocm compile script](../../inference-engine-rocm/scripts/compile_onnx_rocm_docker.sh) — Battle-tested compile script (16 CI iterations)
