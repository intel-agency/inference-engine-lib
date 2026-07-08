# Inference Engine — Review Fix Plan

**Repo:** `intel-agency/inference-engine-lib`
**Target branch:** `development`
**Scope of reviewed work:** the merged diagnostics / MIGraphX / streaming / out-of-process changes on `development` — `InferenceEngineInfo` diagnostics trail + MIGraphX EP migration, streaming autoregressive base, text embedding/generation engines, out-of-process worker IPC, a CI quota fix, and a `buildTransitive` packaging target.

This document is organized so an implementation agent can be pointed at a specific **Section** (High / Medium / Low-Nit / Test-Coverage Gaps). Each item is self-contained: file, problem, fix, and acceptance criteria.

---

## How to use this plan

- **Phase 1** (below) is the priority batch = **High + Medium + Test-Coverage Gaps**. It is enumerated as a single ordered worklist.
- **Sections** retain the full detail for every finding so an agent can be instructed to "fix Section: High" or "fix Section: Test-Coverage Gaps".
- **Creating the PR:** branch off `development`, implement, and open a **new PR against `development`** (the reviewed work is already merged there — do not amend it). Fixes then flow `development → main → release`. Example:
  ```bash
  git checkout development && git pull
  git checkout -b fix/post-review-phase1
  # ... implement, build, test ...
  git push -u origin fix/post-review-phase1
  gh pr create --base development \
    --title "fix: post-review Phase 1 (concurrency, timeouts, decode-delta, disposal)" \
    --body-file docs/review-fix-plan.md
  ```
- **IDs** are stable: `H#` (High), `M#` (Medium), `L#` (Low/Nit), `T#` (Test gap).
- A verified **false alarm** from the initial review is recorded at the end (no action needed) so it is not re-introduced by mistake.

---

## Phase 1 — Priority Worklist (High + Medium + Test-Coverage Gaps)

Ordered by dependency / blast radius. T-items are interleaved near the feature they protect.

| # | ID | Title | Files |
|---|----|-------|-------|
| 1 | H1 | OOP engine: add in-flight guard to `PredictAsync`/`LoadAsync` | `OutOfProcessBaseInferenceEngine.cs` |
| 2 | H2 | OOP engine: make load/predict timeouts reliable (write timeout + kill-on-timeout) | `OutOfProcessBaseInferenceEngine.cs` |
| 3 | M6 | OOP engine: make `Dispose` take the in-flight lock (depends on H1) | `OutOfProcessBaseInferenceEngine.cs` |
| 4 | H5 | `BaseInferenceEngine.LoadAsync`: serialize with a lock | `BaseInferenceEngine.cs` |
| 5 | M4 | `OnSessionInitialized`: fire before warmup; don't leave `_isLoaded` true on warmup throw | `BaseInferenceEngine.cs` |
| 6 | H4 | Streaming: dispose prior KV-carry state; define disposable carry contract | `BaseStreamingInferenceEngine.cs` |
| 7 | H3 | `TextGenerationEngine`: fix decode-delta (no substring-by-length) | `TextGenerationEngine.cs` |
| 8 | M3 | Remove/region-flag dead `SamplingParams`, or implement it in `SampleNextToken` | `StreamingOptions.cs`, `BaseStreamingInferenceEngine.cs` |
| 9 | M1 | `buildTransitive`: derive ROCm lib path from `$(Pkg...)` instead of hardcoded version | `buildTransitive/InferenceEngine.Core.targets`, `InferenceEngine.Core.csproj` |
| 10 | M2 | `ProviderOptions`: populate from real EP options, or rename + document as reserved | `BaseInferenceEngine.cs` |
| 11 | M5 | Decide/document standalone-Core MIGraphX-lib asymmetry | `InferenceEngine.Core.csproj` |
| 12 | M7 | `MaxFrameBytes`: lower cap / stream-deserialize to avoid 256 MiB up-front alloc | `Worker/WorkerProtocol.cs` |
| 13 | T1 | Test native-recoverable fallback branch (`NativeLibraryLoadFailed`) | `EngineInfoTests.cs` |
| 14 | T2 | Test `ProviderAppendFailed` path | `EngineInfoTests.cs` |
| 15 | T3 | Test KV-carry / `CloneAsPastKeyValue` + prior-state disposal (depends on H4) | `StreamingInferenceEngineTests.cs` |
| 16 | T4 | Offline test for `TextGenerationEngine` decode-delta (depends on H3) | `TextEnginesTests.cs` |
| 17 | T5 | Test enumerator-abandoned-without-`DisposeAsync` releases guard | `StreamingInferenceEngineTests.cs` |
| 18 | T6 | Document/wire self-hosted ROCm runner as a required check | `.github/workflows/build-test.yml`, README |

**Exit criteria for Phase 1:** all H/M items resolved, all T-items landed or explicitly deferred with a tracking issue, `dotnet build` 0 warnings/0 errors, full test suite green on all three CI platforms.

---

# Sections

## Section: High

### H1 — OOP engine has no in-flight guard → silent wrong-output corruption
- **File:** `InferenceEngine.Core/OutOfProcessBaseInferenceEngine.cs` (`LoadAsync` ~line 2764, `PredictAsync` ~line 2800)
- **Problem:** Two concurrent `PredictAsync` calls interleave `WriteAsync`/`ReadAsync` on the same stdin/stdout pipe. Request A's write can be followed by B's before A reads, so the host pairs B's request with A's response and returns **wrong inference output with no exception**. The auto-load branch (`if (!_workerLoaded) await LoadAsync()`) compounds it: two callers both see `false` and both call `StartWorker`, killing/spawning the worker twice.
- **Fix:** Serialize the write→read pair with a `SemaphoreSlim(1,1)` (await inside the lock). Set `_workerLoaded` under that lock. Reject reentrant predict-while-loading with `InvalidOperationException`.
- **Acceptance:** A unit/integration test with two concurrent `PredictAsync` calls returns each caller its own correct result (no cross-pairing); no double worker spawn.

### H2 — OOP timeouts are unreliable → host can hang indefinitely
- **File:** `InferenceEngine.Core/OutOfProcessBaseInferenceEngine.cs` (`WriteAsync` ~line 2869, `ReadAsync` ~line 2875)
- **Problem:** `WriteAsync` passes `CancellationToken.None`; if the worker wedges before draining stdin the host blocks forever (`InferenceTimeout` never applies). `ReadAsync` relies on `Stream.ReadAsync(ct)` honoring cancellation, which redirected process stdout pipes historically do **not** do promptly, so a hung worker can hold the host far past `InferenceTimeout`.
- **Fix:** Race `ReadAsync` against `Task.Delay(timeout)` with `Task.WhenAny`; on the timeout branch, `Kill()` the worker and dispose `_fromWorker` to unblock the pending read. Add a `CancellationTokenSource(timeout)` into `WriteAsync` and kill the worker on expiry.
- **Acceptance:** A test that starts a worker which never responds causes the host to throw `TimeoutException` within `StartupTimeout`/`InferenceTimeout` (not hang), and leaves no orphaned process.

### H3 — `TextGenerationEngine` decode-delta assumes prefix-monotonic detokenization
- **File:** `InferenceEngine.Core/Text/TextGenerationEngine.cs` (`ProcessStepOutputs` ~lines 3553-3557)
- **Problem:** The delta logic (`fullDecoded.Substring(gen.DecodedLength)`) assumes `Decode(ids[0..n])` is always a prefix of `Decode(ids[0..n+1])`. This does **not** hold for SentencePiece byte-fallback/normalizers (Phi-3): appending one id can mutate a previously-emitted replacement char, so the prefix shrinks/changes. Net effect: streamed chunks can be **wrong, duplicated, or silently dropped** (the `>` guard masks data loss as `string.Empty`), and `BuildFinalOutput` diverges from a one-shot `Decode`. The existing parity test is network-gated and skips on CI.
- **Fix:** Do not slice by decoded-text length. Either (a) decode only the new id suffix each step with a leading-space reconciliation buffer (canonical HF "delta decode"), or (b) keep a rolling full `Decode` and emit the longest-common-prefix delta against the previous full decode. Add an offline test feeding an id sequence known to mutate its prefix.
- **Acceptance:** Offline unit test (no network/model) demonstrates correct delta emission for a prefix-mutating id sequence; one-shot `==` `string.Concat(stream)` holds on that sequence.

### H4 — KV-cache carry tensors are never disposed → memory growth over long generations
- **File:** `InferenceEngine.Core/BaseStreamingInferenceEngine.cs` (`state = nextState;` ~line 2207; `finally` ~line 2213-2220; `CloneAsPastKeyValue` ~line 2282)
- **Problem:** Each step, `ProcessStepOutputs` returns `nextState` holding freshly-cloned `DenseTensor<float>` KV tensors. On the next iteration `state = nextState;` overwrites the prior reference **without disposing it**, and the `finally` only sets `state = null;`. Over a 256-step run this accumulates ~255 full KV-cache copies in managed memory until GC runs (GC pressure / working-set growth; worse on cancel/exception paths).
- **Fix:** Define the carry-state contract as `IDisposable`. Dispose the previous `state` before reassigning (`if (state is IDisposable d) d.Dispose();`) and dispose `state` in the iterator `finally`. Document that engine-supplied state objects must own/dispose their cloned tensors.
- **Acceptance:** Test (T3) asserts cloned tensors survive output disposal AND the prior step's clone is disposed on step rollover AND final state is disposed on EOS/cancel/exception.

### H5 — `BaseInferenceEngine.LoadAsync` is not thread-safe
- **File:** `InferenceEngine.Core/BaseInferenceEngine.cs` (`_pendingChecks`/`_nextOrder` ~lines 24-25; `_isLoaded` early-return ~line 150; LoadAsync ~line 150)
- **Problem:** `_pendingChecks` (a `List<>`), `_nextOrder`, and the `_isLoaded` early-return guard are mutated on a `Task.Run` thread with no synchronization; `_isLoaded` is a non-volatile read. Two concurrent `LoadAsync` calls can both pass the guard and corrupt the check list / double-construct `_session`. (The `EngineInfo` reference-swap itself is atomic, as documented.)
- **Fix:** Guard `LoadAsync` with a `SemaphoreSlim(1,1)` (async await). Set `_isLoaded` under that lock. Verify the `OnSessionInitialized` "at-most-once" semantics still hold.
- **Acceptance:** A test that awaits two concurrent `LoadAsync` calls does not throw and yields a single consistent `EngineInfo`; the callback fires exactly once.

---

## Section: Medium

### M1 — `buildTransitive` target hardcodes the ROCm lib version path
- **File:** `InferenceEngine.Core/buildTransitive/InferenceEngine.Core.targets` (~line 4167); pin source `InferenceEngine.Core/InferenceEngine.Core.csproj`
- **Problem:** The target hardcodes `inferenceengine.rocm.runtime.linux-x64/1.24.1.36/...`, separately from the `1.24.1.36` `PackageReference`. Because the `<Copy>` is `Condition="Exists(...)"`, a future PackageReference bump without updating this path = **silent no-op**, and consumers hit the exact `OrtSessionOptionsAppendExecutionProvider_MIGraphX` entry-point failure the target exists to prevent.
- **Fix:** Add `GeneratePathProperty="true"` to the `InferenceEngine.ROCm.Runtime.linux-x64` `PackageReference` and use `$(PkgInferenceEngine_ROCm_Runtime_linux_x64)/runtimes/linux-x64/native/libonnxruntime.so` in the target. Remove the hardcoded version.
- **Acceptance:** `grep` for `1.24.1.36` in the targets file returns nothing; bumping the PackageReference in a scratch build changes the resolved path automatically.

### M2 — `ProviderOptions` is a non-functional stub with a misleading test
- **File:** `InferenceEngine.Core/BaseInferenceEngine.cs` (`ExtractProviderOptions` ~line 580); test `InferenceEngine.Tests/EngineInfoTests.cs` (`EngineInfo_ProviderOptions_ReflectsMIGraphXOptions_WhenMIGraphXActive`)
- **Problem:** `ExtractProviderOptions` discards its arg and always returns `Empty`, so `InferenceEngineInfo.ProviderOptions` never reflects `device_id` (or anything). The test named as if it validates reflection only asserts `Assert.Empty(...)`, so it can never fail and gives false confidence.
- **Fix:** Either populate `ProviderOptions` from the actual EP options (at minimum `device_id` for MIGraphX/DirectML), or rename the property/test and mark the member `[Obsolete]`/document it as reserved.
- **Acceptance:** If implemented, the test asserts the real option keys/values; if deferred, the property/test names no longer imply reflection.

### M3 — `SamplingParams` is dead configuration
- **File:** `InferenceEngine.Core/StreamingOptions.cs` (`SamplingParams` ~lines 3016-3029); `BaseStreamingInferenceEngine.cs` (`SampleNextToken` ~lines 2250-2264); `TextGenerationEngine.cs` (~line 3549)
- **Problem:** `StreamingOptions.Sampling` exposes `Temperature/TopK/TopP/RepetitionPenalty` and the `SampleNextToken` docstring says to use it, but the base `SampleNextToken` is greedy argmax and **never reads `CurrentOptions.Sampling`**. Setting `Temperature=0.7` has no effect — a misleading public API.
- **Fix:** Either implement temperature/top-k/top-p in the base `SampleNextToken` (reading `CurrentOptions?.Sampling`), or remove `SamplingParams` + the doc reference and mark the seam `// TODO`.
- **Acceptance:** No public property silently ignores caller intent; if implemented, a test shows temperature>0 producing non-argmax sampling.

### M4 — `OnSessionInitialized` skipped if warmup throws; `_isLoaded` left true
- **File:** `InferenceEngine.Core/BaseInferenceEngine.cs` (~lines 261-265)
- **Problem:** Ordering is `_isLoaded = true; → PerformWarmup(); → FireOnSessionInitialized(info);`. If `PerformWarmup` throws, the callback never fires (contradicting its doc "fires once after the session is created") and the engine is left `_isLoaded == true` with no completed warmup — callers proceed against half-initialized state.
- **Fix:** Move `FireOnSessionInitialized(info)` immediately after `_isLoaded = true` (before warmup), or wrap warmup so a failure cannot prevent the callback and cannot leave `_isLoaded` true.
- **Acceptance:** Test: a warmup throw still fires `OnSessionInitialized` once with a populated info, and `LoadAsync` rethrows the warmup error.

### M5 — Standalone Core build no longer gets the MIGraphX-capable lib (decide + document)
- **File:** `InferenceEngine.Core/InferenceEngine.Core.csproj` (removed `RestoreUpstreamCpuLibOnLinux` ~lines 2648-2666); `buildTransitive/InferenceEngine.Core.targets`
- **Problem:** Removing `RestoreUpstreamCpuLibOnLinux` means Core's *own* output dir resolves to the vanilla CPU `libonnxruntime.so`, while the `buildTransitive` target force-copies the ROCm lib only for *consumers*. So `dotnet build InferenceEngine.Core.csproj -r linux-x64` run directly on a ROCm host lacks the MIGraphX entry point. Asymmetry between Core-build and consumer-build.
- **Fix:** Confirm intent. If Core should self-test MIGraphX, duplicate the `ForceRocmOnnxRuntimeLibOnLinux` copy as an inline `<Target>` in the csproj (inline targets do not propagate from `buildTransitive`). Otherwise document that running Core standalone requires manual lib placement.
- **Acceptance:** Either the standalone build is MIGraphX-capable, or a README/csproj comment states the limitation.

### M6 — `Dispose` races a concurrent in-flight `PredictAsync` (OOP engine)
- **File:** `InferenceEngine.Core/OutOfProcessBaseInferenceEngine.cs` (`Dispose` ~lines 2891-2929)
- **Problem:** `Dispose` sets `_disposed=true` then disposes the streams while a `PredictAsync` that already passed `ThrowIfDisposed()` can be mid-`WriteAsync`/`ReadAsync` → `ObjectDisposedException` or NRE. Reachable once H1's concurrency is real.
- **Fix:** Take the same in-flight lock introduced in H1 inside `Dispose` before disposing the streams (or register a disposal `CancellationToken` that unblocks pending I/O).
- **Acceptance:** A test that disposes during an in-flight predict does not throw an unexpected `ObjectDisposedException`/NRE.

### M7 — `MaxFrameBytes = 256 MiB` allocates up-front
- **File:** `InferenceEngine.Core/Worker/WorkerProtocol.cs` (`MaxFrameBytes` ~line 4047; `ReadExactAsync` ~line 4075)
- **Problem:** A corrupt length prefix or a buggy/malicious worker forces a single 256 MiB contiguous allocation (LOH, can OOM the host), far above any plausible JSON envelope.
- **Fix:** Lower the cap to a realistic value (e.g. 16 MiB) and/or deserialize via `JsonSerializer.DeserializeAsync(stream, ...)` with a size-gated `Stream` wrapper so the payload is never fully buffered.
- **Acceptance:** A test feeding a corrupt 256 MiB-length prefix rejects the frame with `InvalidDataException` without allocating 256 MiB.

---

## Section: Low / Nit

- **L1 — Duplicate `InternalsVisibleTo`** (`InferenceEngine.Core/InferenceEngine.Core.csproj` lines 79 & 158). Compiles fine; remove the inline one (keep the dedicated `<ItemGroup>`).
- **L2 — Bare `dotnet` line** in `scripts/publish-local.ps1` (~line 10415). Prints CLI help then continues; delete it.
- **L3 — `SessionConstructionFailed` enum value is dead** (`InferenceEngine.Core/Diagnostics/InferenceFallbackReason.cs`). No path assigns it; wire it or remove.
- **L4 — Diagnostics discarded on total failure** (`BaseInferenceEngine.cs` outer `LoadAsync` catch). When the CPU fallback session itself throws, `EngineInfo` stays `Empty` despite a rich `_pendingChecks` trail already built; assign a failure snapshot before rethrowing.
- **L5 — Fail-check descriptions say "succeeded"** (`BaseInferenceEngine.cs` ~lines 1841, 1852, 1889, 1910). E.g. `Fail("migraphx.append", "AppendExecutionProvider_MIGraphX succeeded", ...)`; use a neutral description for both Pass/Fail.
- **L6 — `InferenceEngineInfo.Empty` isn't neutral** (`Diagnostics/InferenceEngineInfo.cs` ~lines 2405, 2420, 2425). `Platform`/`SessionCreatedAt` evaluate once at static init, so the "empty" singleton reports the real host platform + a process-start timestamp. Use `Unknown`/`default`.
- **L7 — Empty-input fallback `ids = new[] { 0 }`** (`TextEmbeddingEngine.cs` ~line 3129, `TextGenerationEngine.cs` ~line 3506). Id 0 is `[PAD]` for MiniLM (degenerate embedding) and unknown for Phi-3 (garbage seed). Throw `ArgumentException` on empty input instead.
- **L8 — Inline pooling in `PostProcessWithContext` duplicates the tested `MeanPool` helper** (`TextEmbeddingEngine.cs` ~lines 3209-3223 vs `:3322-3343`). Call the helper to avoid divergence.
- **L9 — `LoadAsync()` runs before the streaming in-flight guard is acquired** (`BaseStreamingInferenceEngine.cs` ~line 2167). A rejected second stream still pays full model-resolution cost; acquire the guard first.
- **L10 — Orphaned worker if host dies abnormally** (`Worker/InferenceWorkerHost.cs`). No `PR_SET_PDEATHSIG` (Linux) / Job Object (Windows); a `SIGKILL`'d host can leak the GPU-holding worker. Bind the child to a Job Object / `prctl`.
- **L11 — Exception type/stack erased across IPC** (`Worker/InferenceWorkerHost.cs` ~lines 3950, 3962, 3986; host re-throws bare `InvalidOperationException` at `OutOfProcessBaseInferenceEngine.cs` ~lines 2789, 2812). Introduce a `RemoteWorkerException` carrying `RemoteTypeName` + `RemoteStackTrace`.
- **L12 — `WorkerArguments`/`WorkerEnvironment` held by reference** (`OutOfProcessEngineOptions.cs` ~lines 2963, 2971). Caller can mutate after construction, changing a later respawn's args/env; defensive-copy into `ImmutableArray`/`FrozenDictionary`.
- **L13 — Deserialized-null `WorkerEnvelope` → NRE** (`Worker/WorkerProtocol.cs` ~line 4061). A `"null"` JSON body yields null but only the EOF path is null-checked. Add `?? throw new InvalidDataException(...)`.
- **L14 — `IsRocmSessionActive` name now misleading** (public property reports MIGraphX activity). Rename in a follow-up (public → breaking change).
- **L15 — Reflection on private `ConfiguredSessionOptions` is fragile** (`RocmProviderTests.cs` ~lines 5477-5483). Now that `InternalsVisibleTo` is wired (after L1 cleanup), expose an internal test hook instead.

---

## Section: Test-Coverage Gaps

- **T1 — Native-recoverable fallback branch untested.** The `catch ... when (IsRecoverableNativeFailure)` branch — the most complex path — has no test asserting `FallbackReason == NativeLibraryLoadFailed` plus the `session.native-recoverable` + `cpu.fallback-*` checks. *(File: `EngineInfoTests.cs`)*
- **T2 — `ProviderAppendFailed` path untested.** No test exercises `AppendExecutionProvider_MIGraphX`/`_DML`/`_CoreML` throwing after all preconditions pass (e.g. ORT built without that EP). Assert `FallbackReason == ProviderAppendFailed` + `*.append` Failed + `*.session` Skipped. *(File: `EngineInfoTests.cs`)*
- **T3 — `CloneAsPastKeyValue` / KV-carry path untested.** The fake engine returns empty outputs and `null` state, so H4's growth and the carry/dispose contract are never exercised. Add a fake that returns a `present_*` float tensor and asserts: (a) the clone survives output disposal, (b) the prior step's clone is disposed on step rollover, (c) state is disposed on EOS/cancel/exception. *(File: `StreamingInferenceEngineTests.cs`; depends on H4)*
- **T4 — No offline test for `TextGenerationEngine` decode-delta.** The only generation test is network-gated and skips on CI. Add a pure unit test with a fake tokenizer that mutates the prefix to catch H3 deterministically. *(File: `TextEnginesTests.cs`; depends on H3)*
- **T5 — Enumerator-abandoned-without-`DisposeAsync` guard release untested.** Today the in-flight guard would leak if the consumer never completes/drives the enumerator to disposal; test that `CurrentOptions` is reset and the guard is released, or document the contract. *(File: `StreamingInferenceEngineTests.cs`)*
- **T6 — GPU/MIGraphX integration tests always skip in CI.** They are `SkippableFact`-gated on `/dev/kfd` + `libamdhip64.so.7` + `libmigraphx_c.so.3`, none present on GitHub-hosted runners, so a regression in the MIGraphX append path passes CI silently. Document in the workflow/README that GPU-provider tests require a self-hosted ROCm runner, and wire that runner as a required check on `main`. *(Files: `.github/workflows/build-test.yml`, `README.md`)*

---

## Appendix — Verified non-issues (do NOT "fix" these)

- **`IsRecoverableNativeFailure` still handles `TypeInitializationException`.** The initial review flagged a possible regression; verified against the `development` branch source (`BaseInferenceEngine.cs:595-601`): the method unwraps `TypeInitializationException` one level and checks the inner against `OnnxRuntimeException`/`DllNotFoundException`/`BadImageFormatException`. Behavior narrowed from recursive to single-level unwrap — acceptable; no action.
- **`IsRocmRuntimeUsable` removal is not a breaking change.** It was `private` (verified in the diff). The `1.0.0-dev → 1.1.0-dev` minor bump is correct (additive new diagnostics/streaming/OOP API).
- **CI restructure is sound.** Build stage = compile-only fast-fail gate (output discarded); test stage does `restore → build --no-restore → test --no-build`. No redundancy; no broken step.

---

## Notes for the implementation agent

- Work top-to-bottom in the **Phase 1 Worklist**; H1 should land before M6 (lock reuse) and H3/H4 before their T-items (T3/T4).
- Keep changes surgical and additive where possible; this PR is already large.
- After each Section, run `dotnet build` (expect 0 warnings/0 errors under `TreatWarningsAsErrors=true`) and `dotnet test` before moving on.
- Do not commit unless explicitly instructed; surface a summary of changes per Section.
- File paths are repo-relative to `intel-agency/inference-engine-lib`.
