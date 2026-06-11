# Feedback: Engine Info Diagnostics API Plan

**Reviewer:** coding agent
**Date:** 2026-06-11
**Plan doc:** `docs/engine-info-api-plan.md`

---

## 1. Check tables (§A.2) — ordering, gaps, and mapping

### 1.1 The `gpu.user-enabled` check belongs *before* `rid.platform`, not after

The current code (`BaseInferenceEngine.cs:243-248`) checks `UseGpuAcceleration`
**before** any platform dispatch. If GPU is disabled, the method returns a CPU
`SessionOptions` immediately — it never probes the platform at all. The plan's
tables put `rid.platform` at order 0 and `gpu.user-enabled` at order 1, which
implies the platform probe runs first. Two options:

**Option A (match current code):** Swap the order. `gpu.user-enabled` becomes
order 0, `rid.platform` becomes order 1 (and is `Skip`-ed when GPU is disabled).
This preserves the existing short-circuit and means a user who sets
`UseGpuAcceleration = false` sees a clean two-check trail: "GPU disabled → done."

**Option B (match the plan):** Move the GPU check after platform detection in
the code. This gives a more informative trail ("we're on Linux x64 *and* you
disabled GPU") but changes the short-circuit behavior.

Recommendation: **Option A.** The short-circuit is intentional and the simpler
trail is more useful. A user who disabled GPU doesn't need to see platform
checks marked "skipped."

### 1.2 The `*.session` checks cannot live inside `ConfiguredSessionOptions`

The plan's check tables include `rocm.session` (order 5), `dml.session` (order 3),
and `coreml.session` (order 3) as if they happen during session-options
configuration. But in the current code, `InferenceSession` construction happens
at `BaseInferenceEngine.cs:167`, *after* `ConfiguredSessionOptions` returns. The
plan's §A.3 refactor pseudocode shows `ConfiguredSessionOptions` returning
`SessionOptions` — it doesn't construct the session.

This means either:

- The `*.session` checks need to be emitted by `LoadAsync`, not
  `ConfiguredSessionOptions`. The `checks` list would need to be threaded from
  `ConfiguredSessionOptions` back into `LoadAsync` (or stored as a field), and
  `LoadAsync` appends the session-construction check after `new InferenceSession(...)`.
- Or the refactor needs to move session construction *into* the branch methods.

Recommendation: **Thread the checks list through both methods.** Keep
`ConfiguredSessionOptions` responsible for provider-append checks and let
`LoadAsync` append the session-construction check. This matches the existing
code structure and keeps the outer try/catch at lines 163-177 as the safety net
for session construction failures. The `InferenceEngineInfo` is then assembled
in `LoadAsync` after both methods have contributed their checks.

### 1.3 The `IsRecoverableNativeFailure` outer catch is a hidden fallback path not covered by checks

The outer catch at `BaseInferenceEngine.cs:170-177` catches
`OnnxRuntimeException`, `DllNotFoundException`, and `BadImageFormatException`
(and `TypeInitializationException` wrapping those). This is a **third** way the
engine can fall back to CPU, distinct from:
1. The per-branch try/catch inside `ConfiguredSessionOptions` (e.g., line 304).
2. The `IsRocmRuntimeUsable()` pre-check returning false (line 282).

The plan's check tables don't have a check for "outer catch triggered." When
this path fires, the per-branch checks would all show `Passed = true` (the
provider was appended successfully), but the session construction failed with a
native error. The consumer would see "all checks passed" but `IsFallback = true`
— confusing.

Recommendation: Add a check named `session.native-recoverable` (or similar)
that is appended by the outer catch block when it fires. Its `Detail` should
include the original exception type and message. The `FallbackReason` in this
case is `SessionConstructionFailed`.

### 1.4 Missing checks worth adding to v1

| Check | Where | Rationale |
|---|---|---|
| `rocm.provider-so-present` | Linux x64, between `hsa.dlopen` and `rocm.append` | `libonnxruntime_providers_rocm.so` is a separate file from `libhsa-runtime64.so`. The csproj's `RestoreUpstreamCpuLibOnLinux` target restores the CPU `libonnxruntime.so` but leaves the ROCm provider `.so` in place. If that file is missing (e.g., stripped by a publish script — see `scripts/import-performers.ps1:89-93` referenced in §B.5), `AppendExecutionProvider_ROCm` will fail with an opaque native error. A `File.Exists` check on the provider `.so` gives a precise diagnostic. |
| `ort.libonnxruntime-loadable` | All platforms, before any EP append | The symlink workaround in the csproj (lines 139-158) exists because P/Invoke probing is fragile. A `NativeLibrary.TryLoad("onnxruntime.dll", ...)` probe before any EP work would catch the "invalid ELF header" failure mode that the `RemoveWindowsNativeLibsOnNonWindows` target is designed to prevent. This is especially valuable on Linux where the symlink chain is the most brittle. |

### 1.5 Checks to explicitly defer to v2

These were mentioned in the plan's open questions. Agree with deferring:

- **`HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` env-var probe.** These affect
  which GPUs the ROCm runtime sees, but parsing them correctly requires
  understanding the device enumeration that `rocm-smi` would provide. Defer
  until multi-GPU enumeration lands.
- **Driver version / ROCm version.** Would require parsing
  `hsa-runtime` version symbols or shelling out to `rocm-smi`. Out of scope.
- **`/dev/dri/renderD*` enumeration.** Useful for confirming the GPU is
  accessible as a DRM render node, but only matters when `/dev/kfd` is present
  and the HSA runtime loads — the existing checks already gate on those.

### 1.6 Parallelism

No. The checks are inherently sequential: each one is a precondition for the
next. Running `hsa.dlopen` in parallel with `kfd.present` saves negligible time
(microseconds for a `File.Exists` and a `dlopen`) and complicates the
`FallbackReason` mapping, which depends on identifying the *first* failing
check. Keep them sequential.

### 1.7 Failure → FallbackReason mapping correction

The table for Linux x64 says:

> `rocm.session` → `SessionConstructionFailed` (run 6-7)

But check 6 is `cpu.fallback-options` and check 7 is `cpu.fallback-session`.
These are the CPU fallback checks. If the ROCm session construction fails, the
engine attempts a CPU fallback — so running checks 6-7 is correct. However, the
current code's outer catch at line 170 only fires when
`IsRecoverableNativeFailure` is true. A non-recoverable exception (e.g.,
`ArgumentException` from bad provider options) would **not** trigger the CPU
fallback — it would propagate to the outer catch at line 195 and be rethrown.
The check table should note this: if `rocm.session` fails with a
non-recoverable exception, checks 6-7 are `Skip`-ed with reason
"non-recoverable exception; no CPU fallback attempted."

---

## 2. API shape (§A.1)

### 2.1 Record vs. class

Records are the right choice for `InferenceEngineInfo` and
`ProviderDiagnosticCheck`. Both are value-like snapshots. `sealed record` with
`init` properties is correct. No pushback.

### 2.2 `InferenceProvider` enum — add `Cpu` explicitly and consider `None`

The enum has `Cpu = 1` which is good. But `Unknown = 0` is ambiguous: does it
mean "not yet loaded" or "loaded but couldn't determine"? Since
`InferenceEngineInfo.Empty` uses `Provider = Unknown` for the pre-load state,
this is fine — but document it explicitly in the XML doc. The consumer needs to
know that `Unknown` means "LoadAsync has not been called" and not "something
went wrong."

Alternatively, consider renaming `Unknown` to `NotLoaded` for clarity. This is
a minor naming nit.

### 2.3 `InferenceFallbackReason` — missing value for the outer-catch path

As noted in §1.3, the outer catch at `BaseInferenceEngine.cs:170-177` can
trigger a CPU fallback for native loading failures that aren't captured by any
existing `FallbackReason`. The closest is `SessionConstructionFailed`, but that
implies the session was constructed and then failed — whereas the outer catch
fires during `new InferenceSession(...)`. Consider adding:

```
NativeLibraryLoadFailed
```

...to distinguish "the ONNX Runtime native library itself couldn't load" from
"the session was constructed but the provider didn't work."

### 2.4 `Skipped` as a separate bool vs. tri-state

The plan uses `Passed` + `Skipped` as two separate bools. This creates an
invalid state: `Passed = true, Skipped = true` (a check can't be both passed
and skipped). A tri-state enum would be cleaner:

```csharp
public enum CheckResult { Passed, Failed, Skipped }
```

Then `ProviderDiagnosticCheck` has a single `Result` property instead of two
bools. The UI rendering logic becomes a single switch instead of "check
`Skipped` before `Passed`" (which the plan itself flags as a risk in §A.10).

Recommendation: **Use the tri-state enum.** It eliminates the invalid state and
simplifies the consumer code. The plan's own risk register calls out the
two-bool approach as a footgun.

### 2.5 `IInferenceEngineDiagnostics` interface — is it needed in v1?

The plan introduces `IInferenceEngineDiagnostics` with a single `EngineInfo`
property. But `BaseInferenceEngine` already implements `IInferenceEngine<TInput,
TOutput>`, and adding `EngineInfo` directly to the base class (as §A.4 shows)
makes the interface redundant for the only known consumer.

If the goal is to allow consumers to depend on diagnostics without depending on
the full base class, the interface makes sense. But if no consumer will ever
hold a reference typed as `IInferenceEngineDiagnostics` (they'll always have
the concrete engine or `IInferenceEngine<TIn, TOut>`), it's premature
abstraction.

Recommendation: **Defer the interface to v2.** Add `EngineInfo` as a public
property on `BaseInferenceEngine` directly (§A.4 already does this). If a
consumer later needs to abstract over diagnostics, extract the interface then.

### 2.6 `OnSessionInitialized` callback — agree with callback over event

Callback is simpler, avoids `EventHandler<T>` ceremony, and the "caught and
logged" semantics are clear. One addition: the callback should receive the
`InferenceEngineInfo` as a parameter (the plan shows this) but should also be
documented as "invoked at most once per `LoadAsync` call." If `LoadAsync` is
called twice (the second call is a no-op due to `_isLoaded`), the callback
should not fire again. The current `LoadAsync` early-returns on `_isLoaded`
(line 148), so this is already the behavior — just document it.

### 2.7 `OnnxRuntimeVersion` — defer to v2

The plan's open question #1 is correct. Capturing the ORT version requires
either reflection over the native lib or a new API call. The value is low:
consumers can read it from the NuGet package version or from
`typeof(InferenceSession).Assembly.GetName().Version`. Drop it from the v1
record and replace with a `string.Empty` default. Add it back in v2 if there's
a clean API.

### 2.8 `HostPlatform` enum and `HostPlatformDetector` — fine, but make it `internal`

The `HostPlatform` enum and `HostPlatformDetector` are implementation details
of the provider-selection logic. Making them `public` expands the API surface
for no consumer benefit — the consumer gets the platform from
`InferenceEngineInfo.Platform`, not by calling `HostPlatformDetector.Current()`
directly.

Recommendation: Make `HostPlatform` and `HostPlatformDetector` `internal`.
Expose the platform only through `InferenceEngineInfo.Platform`. If a consumer
later needs the detector, make it public then.

Wait — `InferenceEngineInfo.Platform` is of type `HostPlatform`, so the enum
must be public. Scratch that. Keep both public but note in XML doc that
`HostPlatformDetector` is provided for convenience and is not the primary API.

### 2.9 `ProviderDiagnosticCheck` — `Order` should be implicit

The `Order` property is described as "0-based ordinal of evaluation." But if
checks are always added to a `List<ProviderDiagnosticCheck>` in order, the
index in the list *is* the order. Having an explicit `Order` property means it
can drift out of sync with the list position.

Recommendation: Either remove `Order` and use the list index, or keep `Order`
and add a debug assertion that `checks[i].Order == i`. The latter is safer for
display purposes (the consumer can sort by `Order` without trusting list
ordering).

---

## 3. Scope of v1 — what to defer

### 3.1 Defer to v2

| Item | Reason |
|---|---|
| `IInferenceEngineDiagnostics` interface | Premature abstraction; no consumer needs it yet (§2.5). |
| `OnnxRuntimeVersion` field | No clean API to capture it; low value (§2.7). |
| `ort.libonnxruntime-loadable` check | Valuable but adds complexity to the refactor; can be added as a follow-up once the core check infrastructure is in place. |
| `rocm.provider-so-present` check | Same — valuable but can be added incrementally. |
| Multi-GPU enumeration | Explicitly called out as v2 in the plan. Agree. |
| ROCm version probing | Agree. |
| Consumer-side `--diag` CLI and UI panel | Already scoped as Section B / separate PR. Agree. |

### 3.2 Keep in v1

Everything else in Section A. The core types, the check infrastructure, the
`EngineInfo` property, the `OnSessionInitialized` callback, and the tests are
all essential and tightly coupled. Shipping them separately would create more
integration risk than shipping together.

---

## 4. Implementation concerns

### 4.1 The `RocmProviderTests` reflection hack will break

`RocmProviderTests.TestRocmEngine.GetConfiguredSessionOptions()` (line 381-393)
uses reflection to call the private `ConfiguredSessionOptions` method and reads
the `out bool` parameter. The plan changes the signature to
`out InferenceEngineInfo`. The reflection call will need to be updated, and the
test assertions that check `rocmProviderAppended` will need to check
`info.Provider == InferenceProvider.Rocm` instead.

This is a straightforward fix, but it should be called out in the PR
description so reviewers know the test change is intentional.

### 4.2 The `ConfiguredSessionOptions` return type change is larger than the plan suggests

The plan's §A.3 shows:

```csharp
private SessionOptions ConfiguredSessionOptions(out InferenceEngineInfo info)
```

But the current signature is:

```csharp
private SessionOptions ConfiguredSessionOptions(out bool rocmProviderAppended)
```

And the call site at line 165 is:

```csharp
using var sessionOptions = ConfiguredSessionOptions(out var rocmProviderAppended);
_session = new InferenceSession(modelPath, sessionOptions);
IsRocmSessionActive = rocmProviderAppended;
```

The refactor needs to:
1. Change the `out` parameter type.
2. Move `IsRocmSessionActive` assignment to derive from `info.Provider == InferenceProvider.Rocm`.
3. Move `EngineInfo` assignment to after the session is constructed (so the session checks can be included).
4. Handle the case where `ConfiguredSessionOptions` succeeds but `new InferenceSession` fails — the `info` needs to be updated with the session-construction failure.

This is more surgery than the plan implies. Consider extracting a private
`BuildEngineInfo` helper that both `ConfiguredSessionOptions` and the outer
catch can contribute to.

### 4.3 Thread safety of `EngineInfo`

`EngineInfo` is set during `LoadAsync`, which runs on a `Task.Run` thread.
`PredictAsync` also runs on `Task.Run` and reads `_session`. If a consumer
reads `EngineInfo` from the UI thread while `LoadAsync` is still running, they
could see a partially-populated record. Since `InferenceEngineInfo` is a
`record` (immutable once constructed), this is safe *if* the property is
assigned atomically (which it is — reference assignment is atomic in .NET).
But the property should be documented as "may return `InferenceEngineInfo.Empty`
until `LoadAsync` completes."

### 4.4 The `OnSessionInitialized` callback runs inside `Task.Run`

The callback fires on the `Task.Run` thread pool thread, not the caller's
synchronization context. If the consumer's callback touches UI state (e.g.,
updates a view model), they'll need to marshal to the UI thread themselves.
Document this.

---

## 5. Minor nits

- §A.1 `InferenceEngineInfo.OnnxRuntimeVersion` XML doc says "Resolved from
  the loaded native lib (Microsoft.ML.OnnxRuntime)." If deferred, the doc
  should say "Not populated in v1; always `string.Empty`."
- §A.2 Linux x64 table, check 3: the plan says `NativeLibrary.TryLoad` but the
  existing code at `BaseInferenceEngine.cs:384` uses `NativeLibrary.Load` (with
  a catch). Either is fine, but `TryLoad` is cleaner for a probe — agree with
  the plan's choice to switch.
- §A.6 log format: `[InferenceEngine]` prefix is fine but consider using
  structured logging properties instead of string interpolation. The existing
  code uses `$"..."` interpolation in some places (line 186, 260) and structured
  logging in others (line 130, 150). The new log should use structured logging
  per the `Microsoft.Extensions.Logging` best practice:
  `_logger.LogInformation("Loaded provider={Provider} ...", info.Provider, ...)`.
- §A.7 test naming: `EngineInfo_AfterLoadOnLinuxArm64_ProviderIsCpu_FallbackIsUnsupportedPlatform`
  is 74 characters. Consider shortening to
  `EngineInfo_LinuxArm64_CpuWithUnsupportedPlatform`. The
  `MethodName_Scenario_ExpectedResult` convention doesn't require every word.

---

## Summary of recommended changes to the plan

1. **Swap check order:** `gpu.user-enabled` before `rid.platform` (or document
   the intentional change).
2. **Thread checks through `LoadAsync`:** Session-construction checks happen in
   `LoadAsync`, not `ConfiguredSessionOptions`.
3. **Add outer-catch check:** `session.native-recoverable` for the
   `IsRecoverableNativeFailure` path.
4. **Use tri-state `CheckResult` enum** instead of `Passed` + `Skipped` bools.
5. **Defer `IInferenceEngineDiagnostics`** interface to v2.
6. **Defer `OnnxRuntimeVersion`** to v2.
7. **Add `NativeLibraryLoadFailed`** to `InferenceFallbackReason`.
8. **Document non-recoverable exception path** in the failure → FallbackReason
   table (checks 6-7 are skipped, not run).
9. **Call out the `RocmProviderTests` reflection break** in the PR description.
10. **Document callback threading** (runs on `Task.Run` thread, not UI thread).
