# Plan: Expose Loaded-Engine Diagnostics (with Fine-Grained Failure Annotations)

**Status:** Draft for lib-team review
**Target repos:**
- Producer: `intel-agency/inference-engine-lib` (this repo) — Section A is the work
- Consumer: `nam20485/InferenceApp` — Section B is the work, depends on Section A
**Goal:** Make the currently selected inference engine observable — provider, device, platform, fallback state, and a precise trail of which checks passed / failed / were skipped during provider selection.

---

## Why

Today, the only signal a consumer gets from the lib about hardware acceleration is
`BaseInferenceEngine.IsRocmSessionActive` (a `bool`, `internal` to the lib,
`InferenceEngine.Core/BaseInferenceEngine.cs:37`). That is:

1. **Insufficient** — it only signals ROCm, never DirectML / CoreML / CPU-fallback / `use-gpu=false` / multi-engine fleets.
2. **Unactionable on failure** — a `false` collapses four different failure modes (`/dev/kfd` missing, `libhsa-runtime64.so` not dlopen-able, `AppendExecutionProvider_ROCm` threw, `new InferenceSession` threw on the ROCm EP) into one indistinguishable `false`. A consumer that wants to fix the situation (install ROCm runtime, install HSA, check dmesg for KFD errors, etc.) cannot.
3. **Unobservable cross-platform** — on Windows, the DirectML `try/catch → CPU` path (`BaseInferenceEngine.cs:256`) silently demotes to CPU with no log; same for CoreML on macOS (`BaseInferenceEngine.cs:270`). Consumers cannot tell "we got a CPU session on purpose" from "we wanted DirectML and silently lost it."
4. **Untested** — `InferenceEngine.Tests/RocmProviderTests.cs` and `FaceDetectionEngineTests.cs` (96–210) exercise the path, but the *consumer-side* test suite (`nam20485/InferenceApp/Tests/`) has zero provider assertions. A ROCm regression on a real GPU host will pass the consumer's 37 tests.

The plan is to:

- Add a structured, post-session `InferenceEngineInfo` describing the engine that was actually loaded.
- Capture, for every platform branch attempted, an ordered list of `ProviderDiagnosticCheck` records — passed, failed, or skipped — so a failed GPU activation can be diagnosed down to the specific missing file or thrown exception.
- Plumb that info into the consumer (InferenceApp) so it can render it in the UI and the CLI.
- Lock the new behavior in with `Xunit.SkippableFact` tests on every branch.

---

## Section A — Lib changes (intel-agency/inference-engine-lib)

### A.1 New types

All in a new namespace `InferenceEngine.Core.Diagnostics`. Add a folder
`InferenceEngine.Core/Diagnostics/` with one file per type:

#### `InferenceProvider` (enum)
```csharp
public enum InferenceProvider
{
    Unknown = 0,
    Cpu,
    Rocm,
    DirectML,
    CoreML,
}
```

#### `HostPlatform` (enum + static detector)
```csharp
public enum HostPlatform
{
    Unknown = 0,
    LinuxX64,
    LinuxArm64,
    WinX64,
    WinArm64,
    OSX64,
    OSXArm64,
}

public static class HostPlatformDetector
{
    public static HostPlatform Current()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            return RuntimeInformation.ProcessArchitecture == Architecture.X64
                ? HostPlatform.LinuxX64
                : HostPlatform.LinuxArm64;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return RuntimeInformation.ProcessArchitecture == Architecture.X64
                ? HostPlatform.OSX64
                : HostPlatform.OSXArm64;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return RuntimeInformation.ProcessArchitecture == Architecture.X64
                ? HostPlatform.WinX64
                : HostPlatform.WinArm64;
        return HostPlatform.Unknown;
    }
}
```

#### `InferenceFallbackReason` (enum)
```csharp
public enum InferenceFallbackReason
{
    None = 0,                 // Selected the requested provider successfully.
    UserDisabledGpu,          // options.UseGpuAcceleration == false.
    UnsupportedPlatform,      // RID did not match a hardware-accel path (e.g., Linux ARM64).
    PlatformProbeFailed,      // Specific OS check failed (e.g., Windows check on a Linux host).
    KfdDeviceMissing,         // /dev/kfd not present.
    HsaRuntimeUnloadable,     // libhsa-runtime64.so not dlopen-able.
    ProviderAppendFailed,     // AppendExecutionProvider_* threw.
    SessionConstructionFailed,// new InferenceSession(..., sessionOptions) threw.
}
```

#### `ProviderDiagnosticCheck` (record)
```csharp
public sealed record ProviderDiagnosticCheck
{
    /// <summary>Stable dotted identifier (e.g. "rocm.append", "kfd.present"). Tests assert on these.</summary>
    public string Name { get; init; } = string.Empty;

    /// <summary>Human-readable description (e.g. "ROCm execution provider appended to session options").</summary>
    public string Description { get; init; } = string.Empty;

    /// <summary>Grouping for UI (e.g. "Platform", "Provider", "Fallback", "Session").</summary>
    public string Category { get; init; } = string.Empty;

    public bool Passed { get; init; }

    /// <summary>Skipped checks have Passed=false and Detail=skip-reason; Failed checks include the exception message.</summary>
    public bool Skipped { get; init; }

    /// <summary>Detail (exception message, file path tried, expected file name, etc.). Null when not informative.</summary>
    public string? Detail { get; init; }

    /// <summary>0-based ordinal of evaluation; preserves the sequence for display.</summary>
    public int Order { get; init; }

    public static ProviderDiagnosticCheck Pass(string name, string description, string category, int order, string? detail = null) => new()
    {
        Name = name, Description = description, Category = category, Order = order, Passed = true, Detail = detail,
    };
    public static ProviderDiagnosticCheck Fail(string name, string description, string category, int order, string detail) => new()
    {
        Name = name, Description = description, Category = category, Order = order, Passed = false, Detail = detail,
    };
    public static ProviderDiagnosticCheck Skip(string name, string description, string category, int order, string reason) => new()
    {
        Name = name, Description = description, Category = category, Order = order, Passed = false, Skipped = true, Detail = reason,
    };
}
```

#### `InferenceEngineInfo` (record)
```csharp
public sealed record InferenceEngineInfo
{
    public InferenceProvider Provider { get; init; } = InferenceProvider.Unknown;
    public HostPlatform Platform { get; init; } = HostPlatformDetector.Current();
    public int DeviceId { get; init; }
    public bool GpuAccelerationRequested { get; init; }

    /// <summary>True when Provider != Unknown AND FallbackReason == None. False when a fall-back occurred.</summary>
    public bool IsFallback { get; init; }
    public InferenceFallbackReason FallbackReason { get; init; } = InferenceFallbackReason.None;

    /// <summary>Per-provider key/value options actually applied (e.g. ROCm: device_id, miopen_conv_exhaustive_search, arena_extend_strategy).</summary>
    public IReadOnlyDictionary<string, string> ProviderOptions { get; init; }
        = ImmutableDictionary<string, string>.Empty;

    /// <summary>Resolved from the loaded native lib (Microsoft.ML.OnnxRuntime).</summary>
    public string OnnxRuntimeVersion { get; init; } = string.Empty;

    public DateTimeOffset SessionCreatedAt { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>Ordered, comprehensive log of every check performed during provider selection. UI displays all entries.</summary>
    public IReadOnlyList<ProviderDiagnosticCheck> Checks { get; init; } = Array.Empty<ProviderDiagnosticCheck>();

    public static InferenceEngineInfo Empty { get; } = new();
}
```

#### `IInferenceEngineDiagnostics` (interface)
```csharp
public interface IInferenceEngineDiagnostics
{
    /// <summary>Snapshot of the engine's loaded state. Throws InvalidOperationException if LoadAsync has not been called.</summary>
    InferenceEngineInfo EngineInfo { get; }
}
```

### A.2 Per-provider check manifest

`ConfiguredSessionOptions` (currently `BaseInferenceEngine.cs:237-324`) is
refactored to emit checks as it runs. The exact checks per branch:

#### Linux x64 → ROCm branch

| Order | Name | Description | Category | Pass criterion | Detail on fail |
|---|---|---|---|---|---|
| 0 | `rid.platform` | Host is Linux x64 | Platform | `HostPlatformDetector.Current() == LinuxX64` | actual platform |
| 1 | `gpu.user-enabled` | `UseGpuAcceleration` is `true` | Config | `options.UseGpuAcceleration` | "UseGpuAcceleration=false" |
| 2 | `kfd.present` | `/dev/kfd` exists | Provider | `File.Exists("/dev/kfd")` | path attempted |
| 3 | `hsa.dlopen` | `libhsa-runtime64.so` is dlopen-able | Provider | `NativeLibrary.TryLoad("libhsa-runtime64.so", out _)` | library name |
| 4 | `rocm.append` | `AppendExecutionProvider_ROCm` succeeded | Provider | no exception | exception message |
| 5 | `rocm.session` | `InferenceSession` constructed with ROCm EP | Session | no exception | exception message |
| 6 | `cpu.fallback-options` | `CreateCpuOnlySessionOptions()` succeeded | Fallback | no exception | (only if previous failed) |
| 7 | `cpu.fallback-session` | `InferenceSession` constructed with CPU EP | Fallback | no exception | (only if previous failed) |

Failure → `FallbackReason` mapping:

| First failing check | FallbackReason |
|---|---|
| `rid.platform` | `UnsupportedPlatform` (and mark 2-7 as `Skip`) |
| `gpu.user-enabled` | `UserDisabledGpu` (and mark 2-7 as `Skip`) |
| `kfd.present` | `KfdDeviceMissing` (and mark 3-5 as `Skip`, run 6-7) |
| `hsa.dlopen` | `HsaRuntimeUnloadable` (and mark 4-5 as `Skip`, run 6-7) |
| `rocm.append` | `ProviderAppendFailed` (mark 5 as `Skip`, run 6-7) |
| `rocm.session` | `SessionConstructionFailed` (run 6-7) |
| `cpu.fallback-session` | (no fallback possible — rethrow original) |

If check 6 or 7 also fails, the engine constructor must still surface the error
clearly (the existing outer try/catch at `BaseInferenceEngine.cs:163-177`
handles this — keep that behavior).

#### Windows → DirectML branch

| Order | Name | Description | Category | Pass criterion |
|---|---|---|---|---|
| 0 | `rid.platform` | Host is Windows | Platform | `HostPlatform == WinX64 \|\| WinArm64` |
| 1 | `gpu.user-enabled` | `UseGpuAcceleration` is `true` | Config | `options.UseGpuAcceleration` |
| 2 | `dml.append` | `AppendExecutionProvider_DML(DeviceId)` succeeded | Provider | no exception |
| 3 | `dml.session` | `InferenceSession` constructed with DML EP | Session | no exception |
| 4 | `cpu.fallback-options` | CPU options built | Fallback | no exception |
| 5 | `cpu.fallback-session` | CPU `InferenceSession` constructed | Fallback | no exception |

Failure → `FallbackReason`:

| First failing check | FallbackReason |
|---|---|
| `rid.platform` | `UnsupportedPlatform` |
| `gpu.user-enabled` | `UserDisabledGpu` |
| `dml.append` | `ProviderAppendFailed` |
| `dml.session` | `SessionConstructionFailed` |

#### macOS → CoreML branch

| Order | Name | Description | Category | Pass criterion |
|---|---|---|---|---|
| 0 | `rid.platform` | Host is macOS | Platform | `HostPlatform == OSX64 \|\| OSXArm64` |
| 1 | `gpu.user-enabled` | `UseGpuAcceleration` is `true` | Config | `options.UseGpuAcceleration` |
| 2 | `coreml.append` | `AppendExecutionProvider_CoreML(ENABLE_ON_SUBGRAPH)` succeeded | Provider | no exception |
| 3 | `coreml.session` | `InferenceSession` constructed with CoreML EP | Session | no exception |
| 4 | `cpu.fallback-options` | CPU options built | Fallback | no exception |
| 5 | `cpu.fallback-session` | CPU `InferenceSession` constructed | Fallback | no exception |

Same `FallbackReason` mapping as Windows.

### A.3 Refactor of `BaseInferenceEngine.ConfiguredSessionOptions`

Replace the current if/else ladder with a small private state machine that
threads a `List<ProviderDiagnosticCheck> checks` through each branch:

```csharp
private SessionOptions ConfiguredSessionOptions(out InferenceEngineInfo info)
{
    var checks = new List<ProviderDiagnosticCheck>();
    var platform = HostPlatformDetector.Current();
    var reqGpu = _options.UseGpuAcceleration;
    int order = 0;

    // Branch dispatch — each branch returns (SessionOptions, InferenceProvider, FallbackReason?).
    var (options, provider, fallback, deviceId) = platform switch
    {
        HostPlatform.LinuxX64 => TryRocmBranch(checks, reqGpu, ref order),
        HostPlatform.WinX64 or HostPlatform.WinArm64 => TryDirectMlBranch(checks, reqGpu, ref order),
        HostPlatform.OSX64 or HostPlatform.OSXArm64 => TryCoreMlBranch(checks, reqGpu, ref order),
        _ => TryCpuBranch(checks, InferenceFallbackReason.UnsupportedPlatform, reqGpu, ref order),
    };

    info = new InferenceEngineInfo
    {
        Provider = provider,
        Platform = platform,
        DeviceId = deviceId,
        GpuAccelerationRequested = reqGpu,
        IsFallback = fallback != InferenceFallbackReason.None,
        FallbackReason = fallback,
        ProviderOptions = ExtractProviderOptions(options, provider),
        OnnxRuntimeVersion = ResolveOnnxRuntimeVersion(),
        SessionCreatedAt = DateTimeOffset.UtcNow,
        Checks = checks,
    };
    return options;
}
```

Each `TryXxxBranch` returns the `SessionOptions` actually used (GPU or CPU
fallback) and appends to the shared `checks` list. The existing
try/catch outer wrapper at `BaseInferenceEngine.cs:163-177` stays — it becomes
the *third* line of defense after per-check failure capture and the
`CreateCpuOnlySessionOptions` helper.

### A.4 `BaseInferenceEngine` API additions (non-breaking)

```csharp
public abstract class BaseInferenceEngine<TInput, TOutput>
    : IInferenceEngine<TInput, TOutput>, IInferenceEngineDiagnostics, IDisposable
{
    // KEEP for back-compat. Marked [Obsolete] in XML doc, retained in code.
    public bool IsRocmSessionActive { get; private set; }

    // NEW.
    public InferenceEngineInfo EngineInfo { get; private set; } = InferenceEngineInfo.Empty;

    // Set by ConfiguredSessionOptions in both success and fallback paths.
    // _session, _runOptions, IsRocmSessionActive, and EngineInfo are all updated
    // in lockstep inside LoadAsync.
}
```

`IsRocmSessionActive` is **not removed**; deprecation is XML-only. Rationale:
the only known consumer (`nam20485/InferenceApp`) does not read it today, but
the lib's public API contract should not shrink in a minor release.

### A.5 `InferenceEngineOptions` additions (non-breaking)

```csharp
public class InferenceEngineOptions
{
    // ... existing properties unchanged ...

    /// <summary>
    /// Optional callback invoked once, on the LoadAsync thread, after the session is created
    /// (including fallback paths). Receives the fully-populated InferenceEngineInfo.
    /// Exceptions thrown by the callback are caught and logged as warnings; they do not
    /// abort the engine.
    /// </summary>
    public Action<InferenceEngineInfo>? OnSessionInitialized { get; set; }
}
```

`UseGpuAcceleration`, `DeviceId`, `WarmupOnLoad`, `ModelPath`, `ModelProvider`
are unchanged. No new required properties.

### A.6 Logging

Per `AGENTS.md` §error_handling: use `ILogger`, no `Console.WriteLine`.
Add a single info log on session init:

```
[InferenceEngine] Loaded provider={Provider} platform={Platform} device={DeviceId} fallback={IsFallback}({FallbackReason}) checks={Checks.Count}
```

If the callback throws, log a warning and continue. Each failed check with a
non-null `Detail` should be logged at `Warning` level so the engine emits
plain-text diagnostic info even before the consumer wires up a UI.

### A.7 Tests (lib-side, in `InferenceEngine.Tests/`)

Use `Xunit.SkippableFact` per `AGENTS.md`. Naming: `MethodName_Scenario_ExpectedResult`.

**Unit (no GPU/ORT required):**

- `EngineInfo_BeforeLoad_ReturnsEmpty` — `EngineInfo` is `InferenceEngineInfo.Empty` until `LoadAsync` runs.
- `EngineInfo_AfterLoadOnLinuxArm64_ProviderIsCpu_FallbackIsUnsupportedPlatform` — pure platform-detector branch.
- `EngineInfo_AfterLoadWithGpuDisabled_ProviderIsCpu_FallbackIsUserDisabledGpu` — `UseGpuAcceleration=false`.
- `EngineInfo_ProviderOptions_ReflectsRocmOptions_WhenRocmActive` — populates `device_id`, `miopen_conv_exhaustive_search`, `arena_extend_strategy`.

**Integration (require real hardware, SkippableFact):**

- `EngineInfo_AfterLoadOnLinuxX64WithKfd_ProviderIsRocm_AllChecksPassed` — `Skip.If(!File.Exists("/dev/kfd"))`.
- `EngineInfo_AfterLoadOnLinuxX64WithoutKfd_ProviderIsCpu_FallbackIsKfdDeviceMissing` — `Skip.If(File.Exists("/dev/kfd"))` (in CI: skip; on real host: assert).
- `EngineInfo_AfterLoadOnWindows_ProviderIsDirectML_AllChecksPassed` — `Skip.IfNot(RuntimeInformation.IsOSPlatform(Windows))`.
- `EngineInfo_AfterLoadOnWindows_ProviderIsCpu_FallbackIsProviderAppendFailed` — when DirectML `try/catch` triggers.
- `EngineInfo_AfterLoadOnMacOS_ProviderIsCoreML_AllChecksPassed` — `Skip.IfNot(RuntimeInformation.IsOSPlatform(OSX))`.
- `EngineInfo_AfterLoadOnMacOS_ProviderIsCpu_FallbackIsProviderAppendFailed`.

**Check-log content tests (no hardware required, mock-friendly):**

- `Checks_OrderedBy_Order_Ascending` — all checks have monotonically increasing Order.
- `Checks_OnRocmFailure_IncludesKfdCheck_WithPassedFalse_AndSkipAnnotationOnDownstream`.
- `Checks_OnUnsupportedPlatform_AllProviderChecksMarkedSkipped_WithReasonUnsupportedPlatform`.
- `OnSessionInitialized_Callback_FiresOnceWithPopulatedInfo_AndDoesNotPropagateExceptions`.
- `OnSessionInitialized_Callback_Throws_EngineStillUsable_AndWarningLogged` (capture with `ILoggerProvider` test sink).

**Coverage target:** maintain or improve baseline (per `AGENTS.md` §coverage_policy). New code is fully covered if every check is exercised by at least one test path — a coverage report should be attached to the PR.

### A.8 Documentation

- Add a **Diagnostics** section to `docs/inferenceapp-integration-guide.md` with a complete code sample (consumer reads `engine.EngineInfo`; iterates `Checks`; formats via `string.Join`).
- Add XML doc comments on every new public/protected member (per `AGENTS.md` §coding_conventions).
- Update `InferenceEngine.Core/InferenceEngine.Core.csproj` `Description` and any package metadata if needed (no code impact).

### A.9 Versioning / publishing

Per `AGENTS.md` §branch_strategy, the work lands on a personal or
`feature/` branch → `development` → `main` → `release`. The version bump
should be a minor (`1.0.x` → `1.1.0`) because the new public types and the
`OnSessionInitialized` option are additive but they expand the public surface.

The package's csproj still references `Microsoft.ML.OnnxRuntime 1.24.1` and
`InferenceEngine.ROCm.Runtime.linux-x64 1.19.2-dev.28` — no new dependencies
expected; the new code uses only BCL + `Microsoft.Extensions.Logging.Abstractions`.

### A.10 Risks & non-goals

**Risks (mitigation in plan):**
- The refactor touches the heart of `BaseInferenceEngine.ConfiguredSessionOptions` — the most platform-sensitive code in the lib. Mitigation: keep the existing outer try/catch at line 163–177 as a final safety net; add per-platform tests so any regression is caught on the matching CI runner.
- The `OnSessionInitialized` callback runs synchronously on the `LoadAsync` thread — a slow callback stalls model load. Documented; not a regression vs. the existing code path.
- `ProviderDiagnosticCheck.Skipped` overlaps with `Passed = false`; UI code must check `Skipped` *before* `Passed` to render the correct icon.

**Non-goals (v2 follow-ups):**
- Multi-GPU enumeration (consumer would have to call `rocm-smi` / DXGI / etc. directly; not a lib concern).
- ROCm version probing (would require parsing `hsa-runtime` version symbols).
- The consumer-side `--diag` CLI subcommand and engine-info UI panel (see Section B).

---

## Section B — Consumer changes (nam20485/InferenceApp)

This depends on Section A. The consumer plan is included here so the lib team
knows what they're enabling; the consumer work itself is a separate PR by
the InferenceApp team.

### B.1 Surface `EngineInfo` to the UI

Two engines exist in the app:
`OnnxInferenceEngine` (detection, `InferenceApp.UI/Services/OnnxInferenceEngine.cs`)
and `ArcFaceRecognitionEngine` (embeddings,
`InferenceApp.UI/Services/ArcFaceRecognitionEngine.cs`). Each gets its own
`EngineInfo` snapshot.

**New view-model `EngineDiagnosticsViewModel`:**
- `IReadOnlyList<ProviderDiagnosticCheck> DetectionChecks { get; }`
- `IReadOnlyList<ProviderDiagnosticCheck> RecognitionChecks { get; }`
- `InferenceEngineInfo? DetectionInfo { get; }`
- `InferenceEngineInfo? RecognitionInfo { get; }`
- `string DetectionSummary => FormatSummary(DetectionInfo)`
- `string RecognitionSummary => FormatSummary(RecognitionInfo)`
- Auto-refreshes when the underlying engines re-load (e.g., when
  `LazyOnnxInferenceEngine` recreates after dispose at
  `InferenceApp.UI/Services/LazyOnnxInferenceEngine.cs:63-75`).

### B.2 UI placement

Two surfaces, low-friction:

1. **Footer / status bar in `MainWindow.axaml`** — always-visible compact summary:
   `[Detect: ROCm/Device 0] [Embed: CPU/Fallback: KfdDeviceMissing]`
   with color cue (green/amber/red) driven by `IsFallback`.

2. **Diagnostics section in `SettingsView.axaml`** — full per-engine
   `EngineInfo` + the ordered `Checks` list with `✓` / `✗` / `—` glyphs and
   each `Detail` text on its own line. Bound to `EngineDiagnosticsViewModel`.

Recommended: ship both. Footer first (small change), Diagnostics section
follows (more layout work).

### B.3 CLI `--diag inference` subcommand

Add a new CLI mode in `InferenceApp.UI/Program.cs` (parallel to the existing
`import-performers` subcommand at `Program.cs:14-18`):

```
$ dotnet run --project InferenceApp.UI -- --diag inference
== Detection Engine (yolov8n-face.onnx) ==
Provider:        ROCm
Platform:        LinuxX64
Device ID:       0
GPU requested:   True
Fallback:        No
ONNX RT version: 1.24.1
Session loaded:  2026-06-11T14:23:01Z
Provider options:
  device_id                       = 0
  miopen_conv_exhaustive_search   = 1
  arena_extend_strategy           = kSameAsRequested
Checks:
  ✓ [0] Platform:   rid.platform
  ✓ [1] Config:     gpu.user-enabled
  ✓ [2] Provider:   kfd.present
  ✓ [3] Provider:   hsa.dlopen
  ✓ [4] Provider:   rocm.append
  ✓ [5] Session:    rocm.session

== Recognition Engine (arcface-recognition.onnx) ==
Provider:        CPU
...
Fallback reason: KfdDeviceMissing
Checks:
  ✓ [0] Platform:   rid.platform
  ✓ [1] Config:     gpu.user-enabled
  ✗ [2] Provider:   kfd.present
        /dev/kfd (file not found)
  — [3] Provider:   hsa.dlopen (skipped: precondition kfd.present failed)
  — [4] Provider:   rocm.append (skipped: precondition kfd.present failed)
  — [5] Session:    rocm.session (skipped: precondition kfd.present failed)
  ✓ [6] Fallback:   cpu.fallback-options
  ✓ [7] Fallback:   cpu.fallback-session
```

Exit code 0 on success, non-zero if `LoadAsync` throws on every path (no
session at all).

### B.4 Tests (consumer-side, in `nam20485/InferenceApp/Tests/`)

- `EngineDiagnosticsViewModel_AfterEngineLoad_PopulatesInfoFromLazyEngine`
  — load `OnnxInferenceEngine` with a known model, assert `DetectionInfo.Provider` matches expected.
- `EngineDiagnosticsViewModel_AfterEngineReload_RefreshesInfo`
  — drive a `LazyOnnxInferenceEngine` dispose+recreate, assert VM re-emits.
- `OnnxInferenceEngine_EngineInfo_ProviderIsCpu_OnLinuxArm64` (xUnit + SkippableFact; add `Xunit.SkippableFact` package if not already present in `Tests/Tests.csproj`).

No new NuGet packages expected — `Tests` already references
`InferenceEngine.Core 1.0.*` (`Tests/Tests.csproj:21`); bump the floating
version after Section A ships a release with the new API.

### B.5 Untouched on purpose

- `scripts/import-performers.ps1:89-93` currently *removes* `libonnxruntime_providers_rocm.so` from the publish output. With the new diagnostics in place, the importer can self-report "no ROCm provider available" via `--diag` and that strip step becomes a deliberate user choice rather than a hard-coded assumption. **Not changed in this plan** — flagged for a follow-up.

---

## Sequencing

1. Section A.1 (types) — pure additions, no behavior change. Smallest possible PR.
2. Section A.3 (refactor) + A.4 (engine property) + A.5 (option) + A.6 (logging) — one PR; behavior-preserving for all existing callers; the only externally visible change is the new `EngineInfo` property and the callback.
3. Section A.7 (tests) — included in the same PR to prove equivalence on each platform.
4. Section A.8 (docs) — same PR.
5. Publish a `-dev.*` build from `development`. Bump `nam20485/InferenceApp`'s floating `1.0.*` → `1.1.0-dev.*` (or whatever Section A's version policy is — confirm with lib team).
6. Section B.1–B.4 (consumer UI + CLI + tests) — second PR, after consumer upgrades the package.

## Open questions for the lib team

1. **Defer `OnnxRuntimeVersion` capture?** `Microsoft.ML.OnnxRuntime` exposes it via `OrtEnv` or via reflection over the loaded native lib. If it costs a new dependency, drop it from the record and let the consumer read it itself.
2. **Should `EngineInfo.Checks` be mutable internally but exposed as `IReadOnlyList`?** Yes is the recommendation — keep the in-flight `List<>` private, freeze it before assignment to `EngineInfo`. Confirm.
3. **Callback vs. event for `OnSessionInitialized`?** Callback was chosen for simplicity and to avoid `EventHandler<T>` allocations on the hot path. Confirm.
4. **`InferenceProvider` extensibility for future EPs (oneDNN, TensorRT, OpenVINO)?** Plan uses a closed enum for v1. Add new values in v2.
