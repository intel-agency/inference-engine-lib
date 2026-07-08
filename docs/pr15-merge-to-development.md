# PR #15 Merge to Development — Implementation Record

**Status:** ✅ Implemented & merged (2026-07-04)
**Merge commit:** `a9aea18` on `development`
**PR:** [#15](https://github.com/intel-agency/inference-engine-lib/pull/15) (MERGED)
**Result:** `development` now holds the Engine-Info diagnostics API + MIGraphX
execution provider + PR #15 review fixes. `main`/`release` still do NOT (see
Promotion below).

---

## What was actually done

### Code changes (2 commits landed on `development`)
1. **`b4df152` — fix(provider): guard redundant CPU fallback + simplify skip-arg callsites**
   - **Comment [2] (real bug):** Added `provider != InferenceProvider.Cpu` to the outer
     recovery catch `when`-clause in `LoadAsync` (`BaseInferenceEngine.cs`), plus
     `InferenceProvider provider = InferenceProvider.Unknown;` initialization so the
     definite-assignment analysis is satisfied when `ConfiguredSessionOptions` itself
     throws. Previously, when the engine had already fallen back to CPU (e.g.
     `KfdDeviceMissing`) and `InferenceSession` construction then threw a recoverable
     native error, the catch re-entered a redundant CPU fallback, clobbered the specific
     `FallbackReason` with `NativeLibraryLoadFailed`, logged a misleading warning, and
     emitted duplicate checks. Now the original reason is preserved and the exception
     propagates (CPU is the last resort).
   - **Comment [5] (cosmetic):** Dropped the default-`true` `skipXxx` arguments at the
     four `SkipRocmProviderChecks` call sites (kfd/hsa/hip/migraphx paths), keeping only
     the non-default (`false`) args per call site.

2. **`b0a54de` — test(diagnostics): use SkippableFact for ORT-dependent EngineInfoTests**
   - Five `EngineInfoTests` were decorated `[Fact]` but throw `SkipException` via
     `Skip.IfNot(IsNativeOnnxRuntimeAvailable())`. On hosts where the ONNX Runtime native
     library is absent (e.g. `macos-latest` CI runner), `[Fact]` reports that
     `SkipException` as a **failure**, failing the macOS job. Switched to
     `[SkippableFact]` so they SKIP cleanly when ORT is unavailable and still run+assert
     where present. This was a pre-existing latent failure previously masked by the
     artifact-storage-quota CI failure.

### Review threads
- All 7 PR #15 review threads now **resolved** (0 open). Comment [2] replied + resolved
  (`3522598947`); comment [5] auto-resolved after its suggestion was applied.

### Branch / worktree cleanup
- `dev/provider-info` (PR #15 head) — deleted (local + remote).
- `feature/migraphx-runtime-probes` (PR #16 head, closed) — deleted (local + remote).
- `inference-engine-lib-wt-provider-info` worktree — removed.

### Validation
- Build: 0 warnings, 0 errors.
- Tests: 82 total (69 passed, 13 skipped, 0 failed) on Linux x64.
- CI: green on all 3 platforms (ubuntu/windows/macos) — run `28695813935`, `success`.

---

## Deviation from the original plan: the HIP multi-SONAME probe was DISCARDED

The original plan (decision #1) was to **carry** an uncommitted multi-SONAME HIP probe
into PR #15. During implementation this was **reversed and discarded**:

- The probe replaced `TryLoad("libamdhip64.so.7", …)` with a loop trying
  `.so.7 / .so.6 / .so.5 / (unversioned)`.
- On this Debian box it "found" `libamdhip64.so.5` (HIP 5.7), but `.so.5` is
  **ABI-incompatible** with the HIP-7.x-built `libonnxruntime.so`. Loading both into one
  process triggered `LLVM ERROR: Option 'disassemble' registered more than once`, which
  **aborted the test host** (a regression vs. the clean baseline).
- The committed baseline (single `.so.7`) runs clean. Therefore the multi-SONAME change
  was a regression and was **not shipped**. `development` keeps the single `.so.7` probe.

The correct fix for actual GPU inference on this box is **not code** — it is installing the
right ROCm runtimes from the AMD ROCm apt repo (`sudo apt install hip-runtime-amd migraphx`)
to obtain the real `libamdhip64.so.7` + `libmigraphx_c.so.3`. The migration guide
(`docs/inferenceapp-migraphx-migration-guide.md` §7) documents this.

---

## Promotion & package publication

The **Publish NuGet** workflow triggers on push to `development` / `main` / `release`.
Merging PR #15 to `development` **already fired it** and **succeeded**, publishing to the
intel-agency GitHub Packages feed:

- **Published version:** `InferenceEngine.Core` **`1.0.0-dev.11`**
  (workflow computed `1.0-dev.11`; NuGet-normalizes to `1.0.0-dev.11`).
- Feed: `https://nuget.pkg.github.com/intel-agency/index.json`
- Versioning scheme: `VERSION_PREFIX=1.0`; `development` → `<prefix>-dev.<run>`,
  `release` → `<prefix>.<run>` (+ NuGet.org only on `release`).

**InferenceApp can consume `1.0.0-dev.11` right now** (no further promotion needed for
testing) by adding the GitHub Packages source + a PAT with `read:packages`, and bumping its
`InferenceEngine.Core` reference to a floating `1.0.0-dev.*` (or pinning `1.0.0-dev.11`).

For a **stable** package version (`1.0.<N>`) + NuGet.org publication, promote
`development → main → release` (PRs per AGENTS.md). The `release` publish additionally
requires the `nuget-org` environment + `NUGET_API_KEY` secret.

---

## Follow-ups (not blocking)
- Promote `development → main → release` if a stable/NuGet.org package is wanted.
- Install AMD ROCm runtimes (`hip-runtime-amd migraphx`) on this Debian box to activate
  MIGraphX GPU inference (verify `EngineInfo.Provider == MIGraphX`).
- Section B: surface `EngineInfo` in InferenceApp (separate repo `nam20485/InferenceApp`).
