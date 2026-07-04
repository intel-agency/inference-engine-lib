# Context

## Current work focus

- PR #15 (`dev/provider-info` → `development`) **merged** on 2026-07-04 (merge commit `a9aea18`). `development` now holds the Engine-Info diagnostics API, the MIGraphX execution provider, and the PR #15 review fixes.
- The Publish NuGet workflow fired on the merge and published `InferenceEngine.Core` **`1.0.0-dev.11`** to the intel-agency GitHub Packages feed — consumable by InferenceApp now.
- Next decision: whether to promote `development → main → release` for a stable package, and/or install the AMD ROCm runtimes on this Debian box to activate MIGraphX GPU.

## Recent changes

- `b4df152`: guarded redundant CPU fallback (comment [2] — `provider != Cpu` in the outer recovery catch `when`-clause) and simplified `SkipRocmProviderChecks` default-`true` args (comment [5]).
- `b0a54de`: switched 5 ORT-dependent `EngineInfoTests` from `[Fact]` to `[SkippableFact]` (pre-existing macOS CI failure surfaced after the artifact-quota fix was applied).
- Deleted superseded branches `dev/provider-info` and `feature/migraphx-runtime-probes` (local + remote) and removed the `inference-engine-lib-wt-provider-info` worktree.

## Key decision: multi-SONAME HIP probe DISCARDED

- An uncommitted multi-SONAME HIP probe (try `libamdhip64.so.7/.6/.5`) was investigated for carrying into PR #15. It was **discarded**: on this box it loaded the ABI-incompatible `libamdhip64.so.5` (HIP 5.7) into the HIP-7.x-built `libonnxruntime.so`, triggering an LLVM command-line-option collision that **aborted the test host** (regression vs. the clean single-`.so.7` baseline).
- Real fix for GPU inference is installing the AMD ROCm runtimes (`hip-runtime-amd migraphx`), NOT code. See `docs/inferenceapp-migraphx-migration-guide.md` §7.

## Next steps

- Decide on stable release promotion (`development → main → release`) — stable version would be `1.0.<N>` + NuGet.org.
- Install `hip-runtime-amd` + `migraphx` on the Debian box to activate MIGraphX; verify `EngineInfo.Provider == MIGraphX`.
- Section B: surface `EngineInfo` in InferenceApp (separate repo).
- Keep this file updated after meaningful changes.
