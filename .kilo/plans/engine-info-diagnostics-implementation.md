# Implementation Plan: Engine Info Diagnostics API

**Source:** `docs/engine-info-api-plan-v2.md`
**Branch:** `feature/engine-info-diagnostics`
**Target:** `development`
**Version bump:** `1.0.x` → `1.1.0` (minor, additive public surface)

---

## Phase 1 — New types (pure additions, no behavior change)

### 1.1 Create `InferenceEngine.Core/Diagnostics/` folder

Create the directory and add one file per type, all in namespace `InferenceEngine.Core.Diagnostics`.

### 1.2 `InferenceEngine.Core/Diagnostics/InferenceProvider.cs`

```csharp
namespace InferenceEngine.Core.Diagnostics;

/// <summary>Identifies the execution provider used by a loaded inference engine.</summary>
public enum InferenceProvider
{
    /// <summary>LoadAsync has not been called yet, or the engine has been disposed.</summary>
    Unknown = 0,
    Cpu,
    Rocm,
    DirectML,
    CoreML,
}
```

### 1.3 `InferenceEngine.Core/Diagnostics/HostPlatform.cs`

```csharp
using System.Runtime.InteropServices;

namespace InferenceEngine.Core.Diagnostics;

/// <summary>Identifies the host platform and architecture.</summary>
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

/// <summary>
/// Detects the host platform at runtime. Provided for convenience; the primary
/// API for platform information is <see cref="InferenceEngineInfo.Platform"/>.
/// </summary>
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

### 1.4 `InferenceEngine.Core/Diagnostics/InferenceFallbackReason.cs`

```csharp
namespace InferenceEngine.Core.Diagnostics;

/// <summary>Reason the engine fell back to CPU (or <see cref="None"/> if no fallback occurred).</summary>
public enum InferenceFallbackReason
{
    None = 0,
    UserDisabledGpu,
    UnsupportedPlatform,
    PlatformProbeFailed,
    KfdDeviceMissing,
    HsaRuntimeUnloadable,
    ProviderAppendFailed,
    SessionConstructionFailed,
    NativeLibraryLoadFailed,
}
```

### 1.5 `InferenceEngine.Core/Diagnostics/CheckResult.cs`

```csharp
namespace InferenceEngine.Core.Diagnostics;

/// <summary>Outcome of a single diagnostic check.</summary>
public enum CheckResult
{
    Passed,
    Failed,
    Skipped,
}
```

### 1.6 `InferenceEngine.Core/Diagnostics/ProviderDiagnosticCheck.cs`

```csharp
namespace InferenceEngine.Core.Diagnostics;

/// <summary>One step in the provider-selection diagnostic trail.</summary>
public sealed record ProviderDiagnosticCheck
{
    /// <summary>Stable dotted identifier (e.g. "rocm.append", "kfd.present"). Tests assert on these.</summary>
    public string Name { get; init; } = string.Empty;

    /// <summary>Human-readable description.</summary>
    public string Description { get; init; } = string.Empty;

    /// <summary>Grouping for UI (e.g. "Platform", "Provider", "Fallback", "Session").</summary>
    public string Category { get; init; } = string.Empty;

    public CheckResult Result { get; init; }

    /// <summary>Detail (exception message, file path tried, etc.). Null when not informative.</summary>
    public string? Detail { get; init; }

    /// <summary>0-based ordinal of evaluation; must equal the check's index in the parent list.</summary>
    public int Order { get; init; }

    public static ProviderDiagnosticCheck Pass(string name, string description, string category, int order, string? detail = null) => new()
    {
        Name = name, Description = description, Category = category, Order = order, Result = CheckResult.Passed, Detail = detail,
    };

    public static ProviderDiagnosticCheck Fail(string name, string description, string category, int order, string detail) => new()
    {
        Name = name, Description = description, Category = category, Order = order, Result = CheckResult.Failed, Detail = detail,
    };

    public static ProviderDiagnosticCheck Skip(string name, string description, string category, int order, string reason) => new()
    {
        Name = name, Description = description, Category = category, Order = order, Result = CheckResult.Skipped, Detail = reason,
    };
}
```

### 1.7 `InferenceEngine.Core/Diagnostics/InferenceEngineInfo.cs`

```csharp
using System.Collections.Immutable;

namespace InferenceEngine.Core.Diagnostics;

/// <summary>Immutable snapshot of a loaded inference engine's provider, platform, and diagnostic trail.</summary>
public sealed record InferenceEngineInfo
{
    public InferenceProvider Provider { get; init; } = InferenceProvider.Unknown;
    public HostPlatform Platform { get; init; } = HostPlatformDetector.Current();
    public int DeviceId { get; init; }
    public bool GpuAccelerationRequested { get; init; }

    /// <summary>True when a fall-back occurred (FallbackReason != None).</summary>
    public bool IsFallback { get; init; }
    public InferenceFallbackReason FallbackReason { get; init; } = InferenceFallbackReason.None;

    /// <summary>Per-provider key/value options actually applied.</summary>
    public IReadOnlyDictionary<string, string> ProviderOptions { get; init; }
        = ImmutableDictionary<string, string>.Empty;

    /// <summary>Not populated in v1; always <see cref="string.Empty"/>.</summary>
    public string OnnxRuntimeVersion { get; init; } = string.Empty;

    public DateTimeOffset SessionCreatedAt { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>Ordered diagnostic trail of every check performed during provider selection.</summary>
    public IReadOnlyList<ProviderDiagnosticCheck> Checks { get; init; } = Array.Empty<ProviderDiagnosticCheck>();

    public static InferenceEngineInfo Empty { get; } = new();
}
```

### 1.8 Verify Phase 1

```bash
dotnet build InferenceEngine.Core
```

No behavior change; all existing tests must still pass.

---

## Phase 2 — Refactor `BaseInferenceEngine` to emit diagnostic checks

### 2.1 Add private field and helpers to `BaseInferenceEngine.cs`

Add these private members:

```csharp
using InferenceEngine.Core.Diagnostics;
using System.Collections.Immutable;
using System.Diagnostics;

// Field — populated during LoadAsync, consumed to build InferenceEngineInfo.
private List<ProviderDiagnosticCheck> _pendingChecks = new();
private int _nextOrder = 0;

private int NextOrder() => _nextOrder++;
```

### 2.2 Add `EngineInfo` public property

```csharp
/// <summary>
/// Snapshot of the engine's loaded state. Returns <see cref="InferenceEngineInfo.Empty"/>
/// until <see cref="LoadAsync"/> completes. Thread-safe: reference assignment is atomic
/// and <see cref="InferenceEngineInfo"/> is an immutable record.
/// </summary>
public InferenceEngineInfo EngineInfo { get; private set; } = InferenceEngineInfo.Empty;
```

### 2.3 Add private helper methods

```csharp
private void AppendSessionCheck(InferenceProvider provider, bool passed, string? detail = null)
{
    var name = provider switch
    {
        InferenceProvider.Rocm => "rocm.session",
        InferenceProvider.DirectML => "dml.session",
        InferenceProvider.CoreML => "coreml.session",
        _ => "cpu.session",
    };
    var desc = $"{provider} InferenceSession construction";
    _pendingChecks.Add(passed
        ? ProviderDiagnosticCheck.Pass(name, desc, "Session", NextOrder(), detail)
        : ProviderDiagnosticCheck.Fail(name, desc, "Session", NextOrder(), detail ?? "unknown failure"));
}

private void SkipRemainingChecks(string reason)
{
    // Skip cpu.fallback-options and cpu.fallback-session
    _pendingChecks.Add(ProviderDiagnosticCheck.Skip("cpu.fallback-options", "CreateCpuOnlySessionOptions()", "Fallback", NextOrder(), reason));
    _pendingChecks.Add(ProviderDiagnosticCheck.Skip("cpu.fallback-session", "CPU InferenceSession construction", "Fallback", NextOrder(), reason));
}

private void AppendFallbackChecks(SessionOptions cpuOptions, string modelPath)
{
    _pendingChecks.Add(ProviderDiagnosticCheck.Pass("cpu.fallback-options", "CreateCpuOnlySessionOptions() succeeded", "Fallback", NextOrder()));
    try
    {
        _session = new InferenceSession(modelPath, cpuOptions);
        _pendingChecks.Add(ProviderDiagnosticCheck.Pass("cpu.fallback-session", "CPU InferenceSession constructed", "Fallback", NextOrder()));
    }
    catch (Exception ex)
    {
        _pendingChecks.Add(ProviderDiagnosticCheck.Fail("cpu.fallback-session", "CPU InferenceSession construction", "Fallback", NextOrder(), ex.Message));
        throw;
    }
}

private void FireOnSessionInitialized(InferenceEngineInfo info)
{
    if (_options.OnSessionInitialized is null) return;
    try
    {
        _options.OnSessionInitialized(info);
    }
    catch (Exception ex)
    {
        _logger.LogWarning(ex, "OnSessionInitialized callback threw; engine is still usable.");
    }
}

private static IReadOnlyDictionary<string, string> ExtractProviderOptions(InferenceProvider provider)
{
    // ROCm options are hardcoded in TryRocmBranch; mirror them here.
    if (provider == InferenceProvider.Rocm)
        return ImmutableDictionary<string, string>.Empty
            .Add("miopen_conv_exhaustive_search", "1")
            .Add("arena_extend_strategy", "kSameAsRequested");
    return ImmutableDictionary<string, string>.Empty;
}
```

### 2.4 Refactor `ConfiguredSessionOptions` signature and body

Change from:
```csharp
private SessionOptions ConfiguredSessionOptions(out bool rocmProviderAppended)
```

To:
```csharp
private SessionOptions ConfiguredSessionOptions(
    out InferenceProvider provider,
    out InferenceFallbackReason fallback,
    out int deviceId)
```

The body dispatches to private branch methods. Each branch method appends checks to `_pendingChecks`.

#### GPU-disabled short-circuit (order 0)

```csharp
if (!_options.UseGpuAcceleration)
{
    _pendingChecks.Add(ProviderDiagnosticCheck.Pass("gpu.user-enabled", "UseGpuAcceleration is true", "Config", NextOrder()));
    // Skip all remaining checks
    _pendingChecks.Add(ProviderDiagnosticCheck.Skip("rid.platform", "Host platform detection", "Platform", NextOrder(), "UseGpuAcceleration=false"));
    _pendingChecks.Add(ProviderDiagnosticCheck.Skip("provider.append", "Execution provider append", "Provider", NextOrder(), "UseGpuAcceleration=false"));
    _pendingChecks.Add(ProviderDiagnosticCheck.Skip("session.construct", "Session construction", "Session", NextOrder(), "UseGpuAcceleration=false"));

    provider = InferenceProvider.Cpu;
    fallback = InferenceFallbackReason.UserDisabledGpu;
    deviceId = 0;
    var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
    options.AppendExecutionProvider_CPU();
    return options;
}
```

#### Platform dispatch

```csharp
var platform = HostPlatformDetector.Current();
_pendingChecks.Add(ProviderDiagnosticCheck.Pass("gpu.user-enabled", "UseGpuAcceleration is true", "Config", NextOrder()));

var (sessionOptions, prov, fb, dev) = platform switch
{
    HostPlatform.LinuxX64 => TryRocmBranch(),
    HostPlatform.WinX64 or HostPlatform.WinArm64 => TryDirectMlBranch(),
    HostPlatform.OSX64 or HostPlatform.OSXArm64 => TryCoreMlBranch(),
    _ => TryCpuBranch(InferenceFallbackReason.UnsupportedPlatform),
};

provider = prov;
fallback = fb;
deviceId = dev;
return sessionOptions;
```

### 2.5 Implement `TryRocmBranch()`

```csharp
private (SessionOptions Options, InferenceProvider Provider, InferenceFallbackReason Fallback, int DeviceId) TryRocmBranch()
{
    _pendingChecks.Add(ProviderDiagnosticCheck.Pass("rid.platform", "Host is Linux x64", "Platform", NextOrder()));
    var deviceId = _options.DeviceId;

    if (!File.Exists("/dev/kfd"))
    {
        _pendingChecks.Add(ProviderDiagnosticCheck.Fail("kfd.present", "/dev/kfd exists", "Provider", NextOrder(), "/dev/kfd"));
        SkipProviderChecks("precondition kfd.present failed");
        return BuildCpuFallback(InferenceFallbackReason.KfdDeviceMissing, deviceId);
    }
    _pendingChecks.Add(ProviderDiagnosticCheck.Pass("kfd.present", "/dev/kfd exists", "Provider", NextOrder()));

    if (!NativeLibrary.TryLoad("libhsa-runtime64.so", typeof(InferenceSession).Assembly,
            DllImportSearchPath.UseDllDirectoryForDependencies | DllImportSearchPath.System32, out var hsaHandle))
    {
        _pendingChecks.Add(ProviderDiagnosticCheck.Fail("hsa.dlopen", "libhsa-runtime64.so is dlopen-able", "Provider", NextOrder(), "libhsa-runtime64.so"));
        SkipProviderChecks("precondition hsa.dlopen failed");
        return BuildCpuFallback(InferenceFallbackReason.HsaRuntimeUnloadable, deviceId);
    }
    if (hsaHandle != IntPtr.Zero) NativeLibrary.Free(hsaHandle);
    _pendingChecks.Add(ProviderDiagnosticCheck.Pass("hsa.dlopen", "libhsa-runtime64.so is dlopen-able", "Provider", NextOrder()));

    var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
    try
    {
        using var rocmOptions = new OrtROCMProviderOptions();
        rocmOptions.UpdateOptions(new Dictionary<string, string>
        {
            ["device_id"] = deviceId.ToString(),
            ["miopen_conv_exhaustive_search"] = "1",
            ["arena_extend_strategy"] = "kSameAsRequested",
        });
        options.AppendExecutionProvider_ROCm(rocmOptions);
        _pendingChecks.Add(ProviderDiagnosticCheck.Pass("rocm.append", "AppendExecutionProvider_ROCm succeeded", "Provider", NextOrder()));
        return (options, InferenceProvider.Rocm, InferenceFallbackReason.None, deviceId);
    }
    catch (Exception ex)
    {
        _pendingChecks.Add(ProviderDiagnosticCheck.Fail("rocm.append", "AppendExecutionProvider_ROCm succeeded", "Provider", NextOrder(), ex.Message));
        _pendingChecks.Add(ProviderDiagnosticCheck.Skip("rocm.session", "ROCm InferenceSession construction", "Session", NextOrder(), "precondition rocm.append failed"));
        options.Dispose();
        return BuildCpuFallback(InferenceFallbackReason.ProviderAppendFailed, deviceId);
    }
}

private void SkipProviderChecks(string reason)
{
    _pendingChecks.Add(ProviderDiagnosticCheck.Skip("hsa.dlopen", "libhsa-runtime64.so is dlopen-able", "Provider", NextOrder(), reason));
    _pendingChecks.Add(ProviderDiagnosticCheck.Skip("rocm.append", "AppendExecutionProvider_ROCm succeeded", "Provider", NextOrder(), reason));
    _pendingChecks.Add(ProviderDiagnosticCheck.Skip("rocm.session", "ROCm InferenceSession construction", "Session", NextOrder(), reason));
}
```

### 2.6 Implement `TryDirectMlBranch()`

```csharp
private (SessionOptions Options, InferenceProvider Provider, InferenceFallbackReason Fallback, int DeviceId) TryDirectMlBranch()
{
    _pendingChecks.Add(ProviderDiagnosticCheck.Pass("rid.platform", "Host is Windows", "Platform", NextOrder()));
    var deviceId = _options.DeviceId;
    var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };

    try
    {
        options.AppendExecutionProvider_DML(deviceId);
        _pendingChecks.Add(ProviderDiagnosticCheck.Pass("dml.append", "AppendExecutionProvider_DML succeeded", "Provider", NextOrder()));
        return (options, InferenceProvider.DirectML, InferenceFallbackReason.None, deviceId);
    }
    catch (Exception ex)
    {
        _pendingChecks.Add(ProviderDiagnosticCheck.Fail("dml.append", "AppendExecutionProvider_DML succeeded", "Provider", NextOrder(), ex.Message));
        _pendingChecks.Add(ProviderDiagnosticCheck.Skip("dml.session", "DirectML InferenceSession construction", "Session", NextOrder(), "precondition dml.append failed"));
        options.Dispose();
        return BuildCpuFallback(InferenceFallbackReason.ProviderAppendFailed, deviceId);
    }
}
```

### 2.7 Implement `TryCoreMlBranch()`

```csharp
private (SessionOptions Options, InferenceProvider Provider, InferenceFallbackReason Fallback, int DeviceId) TryCoreMlBranch()
{
    _pendingChecks.Add(ProviderDiagnosticCheck.Pass("rid.platform", "Host is macOS", "Platform", NextOrder()));
    var deviceId = _options.DeviceId;
    var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };

    try
    {
        options.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
        _pendingChecks.Add(ProviderDiagnosticCheck.Pass("coreml.append", "AppendExecutionProvider_CoreML succeeded", "Provider", NextOrder()));
        return (options, InferenceProvider.CoreML, InferenceFallbackReason.None, deviceId);
    }
    catch (Exception ex)
    {
        _pendingChecks.Add(ProviderDiagnosticCheck.Fail("coreml.append", "AppendExecutionProvider_CoreML succeeded", "Provider", NextOrder(), ex.Message));
        _pendingChecks.Add(ProviderDiagnosticCheck.Skip("coreml.session", "CoreML InferenceSession construction", "Session", NextOrder(), "precondition coreml.append failed"));
        options.Dispose();
        return BuildCpuFallback(InferenceFallbackReason.ProviderAppendFailed, deviceId);
    }
}
```

### 2.8 Implement `TryCpuBranch()` and `BuildCpuFallback()`

```csharp
private (SessionOptions Options, InferenceProvider Provider, InferenceFallbackReason Fallback, int DeviceId) TryCpuBranch(InferenceFallbackReason reason)
{
    _pendingChecks.Add(ProviderDiagnosticCheck.Fail("rid.platform", "Host platform detection", "Platform", NextOrder(), $"Unsupported platform: {HostPlatformDetector.Current()}"));
    _pendingChecks.Add(ProviderDiagnosticCheck.Skip("provider.append", "Execution provider append", "Provider", NextOrder(), "unsupported platform"));
    _pendingChecks.Add(ProviderDiagnosticCheck.Skip("session.construct", "Session construction", "Session", NextOrder(), "unsupported platform"));
    var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
    options.AppendExecutionProvider_CPU();
    return (options, InferenceProvider.Cpu, reason, 0);
}

private (SessionOptions Options, InferenceProvider Provider, InferenceFallbackReason Fallback, int DeviceId) BuildCpuFallback(InferenceFallbackReason reason, int deviceId)
{
    var options = CreateCpuOnlySessionOptions();
    return (options, InferenceProvider.Cpu, reason, deviceId);
}
```

### 2.9 Refactor `LoadAsync`

Replace the existing `LoadAsync` body with the new structure that:
1. Calls `ConfiguredSessionOptions(out provider, out fallback, out deviceId)`
2. Wraps `new InferenceSession(modelPath, sessionOptions)` in try/catch to emit session checks
3. Handles the `IsRecoverableNativeFailure` outer catch with `session.native-recoverable` check
4. Handles non-recoverable exceptions with `SkipRemainingChecks`
5. Assembles `EngineInfo` from `_pendingChecks.ToArray()`
6. Derives `IsRocmSessionActive` from `provider == InferenceProvider.Rocm`
7. Fires `OnSessionInitialized` callback
8. Uses structured logging

See `docs/engine-info-api-plan-v2.md` §A.3 for the full pseudocode.

### 2.10 Add debug assertion on check ordering

After `EngineInfo = new InferenceEngineInfo { ... Checks = _pendingChecks.ToArray() }`:

```csharp
#if DEBUG
for (int i = 0; i < EngineInfo.Checks.Count; i++)
    Debug.Assert(EngineInfo.Checks[i].Order == i, $"Check order mismatch at index {i}: expected {i}, got {EngineInfo.Checks[i].Order}");
#endif
```

### 2.11 Verify Phase 2

```bash
dotnet build
dotnet test
```

All existing tests must pass. `IsRocmSessionActive` behavior is preserved.

---

## Phase 3 — Update `InferenceEngineOptions`

### 3.1 Add `OnSessionInitialized` callback

In `InferenceEngine.Core/InferenceEngineOptions.cs`, add:

```csharp
using InferenceEngine.Core.Diagnostics;

/// <summary>
/// Optional callback invoked once, on the LoadAsync thread pool thread, after the session
/// is created (including fallback paths). Receives the fully-populated InferenceEngineInfo.
/// Exceptions thrown by the callback are caught and logged as warnings; they do not
/// abort the engine.
/// <para>
/// Invoked at most once per LoadAsync call. If LoadAsync is called again after the engine
/// is already loaded, the callback does not fire.
/// </para>
/// <para>
/// Runs on a Task.Run thread pool thread, not the caller's synchronization context.
/// Consumers that touch UI state must marshal to the UI thread themselves.
/// </para>
/// </summary>
public Action<InferenceEngineInfo>? OnSessionInitialized { get; set; }
```

### 3.2 Verify Phase 3

```bash
dotnet build
```

---

## Phase 4 — Migrate existing tests

### 4.1 Update `RocmProviderTests.TestRocmEngine.GetConfiguredSessionOptions()`

The reflection call at `RocmProviderTests.cs:381-393` targets the old signature:
```csharp
private SessionOptions ConfiguredSessionOptions(out bool rocmProviderAppended)
```

Update to target the new signature:
```csharp
private SessionOptions ConfiguredSessionOptions(
    out InferenceProvider provider,
    out InferenceFallbackReason fallback,
    out int deviceId)
```

The reflection invocation changes from:
```csharp
object?[] parameters = [false];
var options = method.Invoke(this, parameters) as SessionOptions;
var rocmProviderAppended = parameters[0] is true;
return (options, rocmProviderAppended);
```

To:
```csharp
object?[] parameters = [InferenceProvider.Unknown, InferenceFallbackReason.None, 0];
var options = method.Invoke(this, parameters) as SessionOptions
    ?? throw new InvalidOperationException("ConfiguredSessionOptions did not return SessionOptions.");
var provider = (InferenceProvider)parameters[0]!;
return (options, provider);
```

Update the return type of `GetConfiguredSessionOptions()` to:
```csharp
public (SessionOptions Options, InferenceProvider Provider) GetConfiguredSessionOptions()
```

### 4.2 Update all test assertions

Replace all `Assert.True(configuredOptions.RocmProviderAppended)` with:
```csharp
Assert.Equal(InferenceProvider.Rocm, configuredOptions.Provider);
```

Replace all `Assert.False(rocmProviderAppended)` with:
```csharp
Assert.NotEqual(InferenceProvider.Rocm, configuredOptions.Provider);
```

Add `using InferenceEngine.Core.Diagnostics;` to the test file.

### 4.3 Verify Phase 4

```bash
dotnet test --filter "FullyQualifiedName~RocmProviderTests"
```

---

## Phase 5 — Add new tests

### 5.1 Create `InferenceEngine.Tests/EngineInfoTests.cs`

New test file with the following tests (all using `Xunit.SkippableFact` where platform-specific):

**Unit (no GPU/ORT required):**
- `EngineInfo_BeforeLoad_ReturnsEmpty`
- `EngineInfo_GpuDisabled_CpuWithUserDisabledGpu`
- `EngineInfo_ProviderOptions_ReflectsRocmOptions_WhenRocmActive`

**Integration (SkippableFact):**
- `EngineInfo_LinuxX64WithKfd_RocmWithAllChecksPassed` — `Skip.If(!File.Exists("/dev/kfd"))`
- `EngineInfo_LinuxX64WithoutKfd_CpuWithKfdDeviceMissing` — `Skip.If(File.Exists("/dev/kfd"))`
- `EngineInfo_Windows_DirectMLWithAllChecksPassed` — `Skip.IfNot(RuntimeInformation.IsOSPlatform(Windows))`
- `EngineInfo_MacOS_CoreMLWithAllChecksPassed` — `Skip.IfNot(RuntimeInformation.IsOSPlatform(OSX))`

**Check-log content tests:**
- `Checks_OrderedByOrder_Ascending`
- `Checks_OnRocmFailure_IncludesKfdCheck_WithResultFailed_AndSkipAnnotationOnDownstream`
- `Checks_OnUnsupportedPlatform_AllProviderChecksMarkedSkipped_WithReasonUnsupportedPlatform`
- `Checks_OnNonRecoverableException_FallbackChecksMarkedSkipped_WithReasonNoRecovery`
- `OnSessionInitialized_Callback_FiresOnceWithPopulatedInfo_AndDoesNotPropagateExceptions`
- `OnSessionInitialized_Callback_Throws_EngineStillUsable_AndWarningLogged`
- `OnSessionInitialized_Callback_SecondLoadAsync_DoesNotFireAgain`

Use the same `TestRocmEngine` pattern from `RocmProviderTests.cs` for the test engine, or create a new `TestEngine` class that exposes `EngineInfo` after `LoadAsync`.

### 5.2 Verify Phase 5

```bash
dotnet test
./scripts/agent-self-check.sh
```

---

## Phase 6 — Documentation

### 6.1 Add Diagnostics section to `docs/inferenceapp-integration-guide.md`

Add a section with:
- Code sample showing consumer reading `engine.EngineInfo`
- Iterating `Checks` and formatting via `string.Join`
- Example output for ROCm success and CPU fallback

### 6.2 Update `InferenceEngine.Core.csproj` description

Update the `<Description>` element to mention diagnostics:
```xml
<Description>A cross-platform .NET library for AI inference with ONNX Runtime, featuring hardware acceleration, automatic fallback, and built-in diagnostics for provider selection.</Description>
```

### 6.3 Verify Phase 6

```bash
dotnet build
```

---

## Phase 7 — Version bump and CI

### 7.1 Update version in `InferenceEngine.Core.csproj`

Change:
```xml
<Version>1.0.0-dev</Version>
```
To:
```xml
<Version>1.1.0-dev</Version>
```

### 7.2 Run full validation

```bash
./scripts/agent-self-check.sh
```

This runs lint, build, and tests. Fix any failures.

### 7.3 Collect coverage

```bash
dotnet test --collect:"XPlat Code Coverage"
```

Verify coverage has not regressed vs. baseline.

### 7.4 Commit and push

```bash
git add -A
git commit -m "feat: add InferenceEngineInfo diagnostics API with per-check provider trail

- Add InferenceEngine.Core.Diagnostics namespace with InferenceEngineInfo,
  ProviderDiagnosticCheck, CheckResult, InferenceProvider, HostPlatform,
  InferenceFallbackReason types
- Refactor BaseInferenceEngine.ConfiguredSessionOptions to emit ordered
  diagnostic checks for each platform branch (ROCm, DirectML, CoreML, CPU)
- Add EngineInfo property to BaseInferenceEngine (non-breaking)
- Add OnSessionInitialized callback to InferenceEngineOptions
- Add session.native-recoverable check for IsRecoverableNativeFailure path
- Use tri-state CheckResult enum (Passed/Failed/Skipped)
- Migrate RocmProviderTests reflection hack to new signature
- Add EngineInfoTests with unit, integration, and check-content tests
- Version bump 1.0.0-dev → 1.1.0-dev"
git push -u origin feature/engine-info-diagnostics
```

### 7.5 Open PR targeting `development`

```bash
gh pr create --base development --title "feat: InferenceEngineInfo diagnostics API" --body "..."
```

### 7.6 Monitor CI

```bash
gh run list --limit 5 --branch feature/engine-info-diagnostics
gh run watch <run-id> --exit-status
```

Fix any CI failures and push follow-up commits.

---

## Files changed summary

| File | Action |
|---|---|
| `InferenceEngine.Core/Diagnostics/InferenceProvider.cs` | Create |
| `InferenceEngine.Core/Diagnostics/HostPlatform.cs` | Create |
| `InferenceEngine.Core/Diagnostics/InferenceFallbackReason.cs` | Create |
| `InferenceEngine.Core/Diagnostics/CheckResult.cs` | Create |
| `InferenceEngine.Core/Diagnostics/ProviderDiagnosticCheck.cs` | Create |
| `InferenceEngine.Core/Diagnostics/InferenceEngineInfo.cs` | Create |
| `InferenceEngine.Core/BaseInferenceEngine.cs` | Modify (add field, property, helpers, refactor `ConfiguredSessionOptions` and `LoadAsync`) |
| `InferenceEngine.Core/InferenceEngineOptions.cs` | Modify (add `OnSessionInitialized`) |
| `InferenceEngine.Core/InferenceEngine.Core.csproj` | Modify (version bump, description update) |
| `InferenceEngine.Tests/RocmProviderTests.cs` | Modify (migrate reflection hack) |
| `InferenceEngine.Tests/EngineInfoTests.cs` | Create |
| `docs/inferenceapp-integration-guide.md` | Modify (add Diagnostics section) |

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `ConfiguredSessionOptions` refactor breaks platform-specific behavior | Keep outer try/catch at lines 163-177 as safety net; per-platform tests on CI |
| `RocmProviderTests` reflection hack fails silently | Update in same PR; call out in PR description |
| `EngineInfo` thread safety | Immutable record + atomic reference assignment; document "may return Empty until LoadAsync completes" |
| `OnSessionInitialized` callback blocks LoadAsync | Document "runs on Task.Run thread"; catch and log exceptions |
| Coverage regression | Run `dotnet test --collect:"XPlat Code Coverage"` and compare to baseline |
