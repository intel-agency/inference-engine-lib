# MIGraphX Rebase + Runtime Dependency Probes

## Goal

1. Rebase uncommitted MIGraphX changes from `dev/downgrade-1_19_2` onto `dev/provider-info`
2. Integrate MIGraphX EP into the diagnostics-aware `TryRocmBranch()`
3. Add runtime dependency probes for `libamdhip64.so.7` and `libmigraphx_c.so.3` with actionable error messages
4. Build, test, verify

## Branch Strategy

```
dev/downgrade-1_19_2  (current, 6 uncommitted files)
  ↓ git stash
dev/provider-info     (checkout)
  ↓ git checkout -b feature/migraphx-runtime-probes
feature/migraphx-runtime-probes  (new branch)
  ↓ git stash pop + resolve conflicts
```

## Phase 1: Rebase

1. `git stash` — save MIGraphX changes
2. `git checkout dev/provider-info`
3. `git checkout -b feature/migraphx-runtime-probes`
4. `git stash pop` — apply changes, resolve conflicts in:
   - `BaseInferenceEngine.cs` — diagnostics version has `TryRocmBranch()`, not the old inline block
   - `InferenceEngine.Core.csproj` — diagnostics version may have different structure
   - `InferenceEngine.Tests.csproj` — same
   - `RocmProviderTests.cs` — diagnostics version uses `TestInfoEngine` not `TestRocmEngine`
   - `FaceDetectionEngineTests.cs` — may conflict

## Phase 2: csproj Changes (same as before)

### Core csproj
- NuGet version: `1.19.2-dev.28` → `1.24.1-dev.33`
- Comment block: MIGraphX narrative (no version mismatch, no SIGSEGV)
- Symlinks: `providers_rocm` → `providers_migraphx`
- Remove `RestoreUpstreamCpuLibOnLinux` target entirely

### Tests csproj
- Symlinks: `providers_rocm` → `providers_migraphx`
- Remove `RestoreUpstreamCpuLibOnLinux` target entirely

## Phase 3: Diagnostics Integration

### 3a. Update `InferenceProvider` enum

```csharp
public enum InferenceProvider
{
    Unknown = 0,
    Cpu,
    MIGraphX,  // was: Rocm
    DirectML,
    CoreML,
}
```

**Decision**: Rename `Rocm` → `MIGraphX` because the enum identifies the EP, not the platform. The platform is already captured by `HostPlatform.LinuxX64`.

### 3b. Add new `InferenceFallbackReason` values

```csharp
public enum InferenceFallbackReason
{
    None = 0,
    UserDisabledGpu,
    UnsupportedPlatform,
    PlatformProbeFailed,
    KfdDeviceMissing,
    HsaRuntimeUnloadable,
    HipRuntimeMissing,      // NEW: libamdhip64.so.7 not found
    MIGraphXRuntimeMissing, // NEW: libmigraphx_c.so.3 not found
    ProviderAppendFailed,
    SessionConstructionFailed,
    NativeLibraryLoadFailed,
}
```

### 3c. Rewrite `TryRocmBranch()` in `BaseInferenceEngine.cs`

Current flow on `dev/provider-info`:
```
kfd.present → hsa.dlopen → rocm.append (OrtROCMProviderOptions)
```

New flow:
```
kfd.present → hsa.dlopen → hip.dlopen → migraphx.dlopen → migraphx.append
```

Each probe uses `NativeLibrary.TryLoad()` and adds a `ProviderDiagnosticCheck` (Pass/Fail/Skip). On failure, the detail string includes:
- What library is missing
- Why it's needed
- Exact install command

Example detail for `hip.dlopen` failure:
```
libamdhip64.so.7 not found. HIP runtime is required for MIGraphX GPU acceleration.
Install: sudo apt install hip-runtime-amd (from AMD ROCm repository)
See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
```

Example detail for `migraphx.dlopen` failure:
```
libmigraphx_c.so.3 not found. MIGraphX runtime is required for GPU-accelerated inference.
Install: sudo apt install migraphx (from AMD ROCm repository)
See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
```

```csharp
private (...) TryRocmBranch()
{
    _pendingChecks.Add(Pass("rid.platform", ...));
    var deviceId = _options.DeviceId;

    // 1. /dev/kfd
    if (!File.Exists("/dev/kfd")) { ... KfdDeviceMissing ... }
    _pendingChecks.Add(Pass("kfd.present", ...));

    // 2. HSA runtime
    if (!NativeLibrary.TryLoad("libhsa-runtime64.so", ..., out var hsaHandle)) { ... HsaRuntimeUnloadable ... }
    NativeLibrary.Free(hsaHandle);
    _pendingChecks.Add(Pass("hsa.dlopen", ...));

    // 3. HIP runtime (NEW)
    if (!NativeLibrary.TryLoad("libamdhip64.so.7", ..., out var hipHandle))
    {
        _pendingChecks.Add(Fail("hip.dlopen", "libamdhip64.so.7 is loadable", "Provider", NextOrder(),
            "libamdhip64.so.7 not found. HIP runtime is required for MIGraphX GPU acceleration. " +
            "Install: sudo apt install hip-runtime-amd (from AMD ROCm repository). " +
            "See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"));
        SkipRemainingRocmChecks("precondition hip.dlopen failed");
        return BuildCpuFallback(InferenceFallbackReason.HipRuntimeMissing, deviceId);
    }
    NativeLibrary.Free(hipHandle);
    _pendingChecks.Add(Pass("hip.dlopen", ...));

    // 4. MIGraphX runtime (NEW)
    if (!NativeLibrary.TryLoad("libmigraphx_c.so.3", ..., out var mgxHandle))
    {
        _pendingChecks.Add(Fail("migraphx.dlopen", "libmigraphx_c.so.3 is loadable", "Provider", NextOrder(),
            "libmigraphx_c.so.3 not found. MIGraphX runtime is required for GPU-accelerated inference. " +
            "Install: sudo apt install migraphx (from AMD ROCm repository). " +
            "See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"));
        SkipRemainingRocmChecks("precondition migraphx.dlopen failed");
        return BuildCpuFallback(InferenceFallbackReason.MIGraphXRuntimeMissing, deviceId);
    }
    NativeLibrary.Free(mgxHandle);
    _pendingChecks.Add(Pass("migraphx.dlopen", ...));

    // 5. MIGraphX EP append
    var options = new SessionOptions { ... };
    try
    {
        options.AppendExecutionProvider_MIGraphX(deviceId);
        _pendingChecks.Add(Pass("migraphx.append", ...));
        return (options, InferenceProvider.MIGraphX, None, deviceId);
    }
    catch (Exception ex)
    {
        _pendingChecks.Add(Fail("migraphx.append", ..., ex.Message));
        options.Dispose();
        return BuildCpuFallback(InferenceFallbackReason.ProviderAppendFailed, deviceId);
    }
}
```

### 3d. Update `SkipRocmProviderChecks` → `SkipRemainingRocmChecks`

Add `hip.dlopen`, `migraphx.dlopen`, `migraphx.append` to the skip list.

### 3e. Update `AppendSessionCheck` provider mapping

```csharp
InferenceProvider.MIGraphX => "migraphx.session",
```

## Phase 4: Test Updates

### 4a. `EngineInfoTests.cs`
- Rename `InferenceProvider.Rocm` → `InferenceProvider.MIGraphX` in all assertions
- Rename test methods: `*Rocm*` → `*MIGraphX*`
- Update `IsRocmRuntimeUsableOnHost()` to also probe for HIP and MIGraphX libs
- Add new tests:
  - `EngineInfo_LinuxX64WithKfd_CpuWithHipRuntimeMissing` — skip if HIP is present
  - `EngineInfo_LinuxX64WithKfd_CpuWithMIGraphXRuntimeMissing` — skip if MIGraphX is present

### 4b. `RocmProviderTests.cs`
- Update `TestRocmEngine` → use diagnostics-aware base (if not already migrated)
- Update skip messages and assertions for MIGraphX

### 4c. `FaceDetectionEngineTests.cs`
- `IsRocmLibPresent()` → `IsMIGraphXLibPresent()` (file path changes)
- Test name: `ShouldUseMIGraphXProvider`
- Assertion message updated

### 4d. `NativeLibraryValidationTests.cs` (integration tests)
- All `*Rocm*` → `*MIGraphX*` renames
- Size threshold lowered for thin plugin .so
- Symbol assertions updated
- EP string: `MIGraphXExecutionProvider`

## Phase 5: Build & Verify

1. `dotnet build` — 0 errors, 0 warnings
2. `dotnet test InferenceEngine.Tests` — all pass/skip as expected
3. Grep verification:
   - `providers_rocm` → zero hits
   - `ROCmExecutionProvider` → zero hits
   - `OrtROCMProviderOptions` → zero hits
   - `AppendExecutionProvider_ROCm` → zero hits
4. Run the failing test with MIGraphX core .so to confirm it now fails with a clear diagnostic message (not a silent CPU fallback)

## Files Changed

| File | Change |
|---|---|
| `InferenceEngine.Core/InferenceEngine.Core.csproj` | NuGet version, comments, symlinks, remove RestoreUpstreamCpuLibOnLinux |
| `InferenceEngine.Core/BaseInferenceEngine.cs` | Rewrite `TryRocmBranch()` with HIP + MIGraphX probes |
| `InferenceEngine.Core/Diagnostics/InferenceProvider.cs` | `Rocm` → `MIGraphX` |
| `InferenceEngine.Core/Diagnostics/InferenceFallbackReason.cs` | Add `HipRuntimeMissing`, `MIGraphXRuntimeMissing` |
| `InferenceEngine.Tests/InferenceEngine.Tests.csproj` | Symlinks, remove RestoreUpstreamCpuLibOnLinux |
| `InferenceEngine.Tests/EngineInfoTests.cs` | Rename Rocm→MIGraphX, add HIP/MIGraphX probe tests |
| `InferenceEngine.Tests/RocmProviderTests.cs` | Update for MIGraphX |
| `InferenceEngine.Tests/FaceDetectionEngineTests.cs` | Helper rename, test rename |
| `InferenceEngine.Core.IntegrationTests/NativeLibraryValidationTests.cs` | All renames + thresholds |

## Estimated Scope

~200 lines changed across 9 files. Mostly mechanical renames + 2 new probe blocks (~30 lines each) in `TryRocmBranch()`.
