# InferenceApp: MIGraphX Migration Guide

**Target:** `nam20485/InferenceApp`  
**Package:** `InferenceEngine.Core` (from `intel-agency/inference-engine-lib`)  
**Scope:** ROCm EP removed → MIGraphX EP, new Diagnostics API, breaking enum changes

---

## Summary of Changes

The upstream `InferenceEngine.Core` library has migrated from the deprecated **ROCm Execution Provider** to **MIGraphX Execution Provider** (ORT 1.24.1). This affects InferenceApp in three areas:

1. **Package version** — bump `InferenceEngine.Core` to `1.24.1-dev.*` (or later)
2. **ORT version** — bump `Microsoft.ML.OnnxRuntime` to `1.24.1` (or later)
3. **Enum values** — `InferenceProvider.Rocm` is now `InferenceProvider.MIGraphX`

The `IInferenceEngine<TInput, TOutput>` contract is **unchanged** — `LoadAsync` and `PredictAsync` signatures are identical. Most of InferenceApp requires **zero code changes**.

---

## Step 1 — Update Package Versions

### `InferenceApp.UI/InferenceApp.UI.csproj`

Current:

```xml
<PackageReference Include="InferenceEngine.Core" Version="1.0.9" />
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.26.0" />
```

After:

```xml
<PackageReference Include="InferenceEngine.Core" Version="1.24.1-dev.*" />
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.24.1" />
```

> **Why downgrade ORT from 1.26.0 to 1.24.1?** The `InferenceEngine.Core` package includes platform-specific
> native libraries compiled against ORT 1.24.1. The managed bindings must match exactly to avoid ABI mismatches.
> ORT 1.26.0 managed bindings are not compatible with ORT 1.24.1 native shared libraries.

### `Tests/Tests.csproj`

Current:

```xml
<PackageReference Include="InferenceEngine.Core" Version="1.0.*" />
```

After:

```xml
<PackageReference Include="InferenceEngine.Core" Version="1.24.1-dev.*" />
```

---

## Step 2 — No Code Changes Required (What Stays the Same)

### Files with zero changes needed

| File | Why no change needed |
|---|---|
| `Services/IInferenceEngine.cs` | UI-facing interface — no reference to `InferenceProvider` or diagnostics |
| `Services/NullInferenceEngine.cs` | Stub implementation — no engine internals |
| `Services/LazyOnnxInferenceEngine.cs` | Wraps `OnnxInferenceEngine` via `IInferenceEngine` — no enum references |
| `Services/VideoProcessorService.cs` | Consumes `IInferenceEngine` abstraction — no provider-specific code |
| `ViewModels/MainViewModel.cs` | Uses `IInferenceEngine` — no provider awareness |
| `ViewModels/SettingsViewModel.cs` | No inference engine references |
| `ViewModels/VideoDetectionViewModel.cs` | UI-only view model |
| `Program.cs` | App startup — no engine internals |
| `Services/AppSettings.cs` | Configuration only — no engine types |
| `Automation/AutomationRunner.cs` | UI automation — no engine internals |

### Why these files don't change

InferenceApp has a clean layered architecture:

```text
ViewModels / Windows
       ↓
 IInferenceEngine (UI contract — byte[] → PredictionResult[])
       ↓
 LazyOnnxInferenceEngine (lazy init wrapper)
       ↓
 OnnxInferenceEngine : BaseInferenceEngine<byte[], List<PredictionResult>>
       ↓
 InferenceEngine.Core (NuGet) — handles EP selection transparently
```

The provider selection logic (`MIGraphX` vs `DirectML` vs `CoreML` vs `CPU`) is entirely inside `BaseInferenceEngine` in the NuGet package. Downstream apps never touch it.

---

## Step 3 — Code That References Provider Names (If Any)

Search InferenceApp for any references to `Rocm` or `InferenceProvider`:

```bash
grep -r "Rocm\|InferenceProvider\|IsRocmSessionActive\|InferenceFallbackReason" \
  --include="*.cs" --include="*.xaml" --include="*.axaml" \
  InferenceApp.UI/ Tests/
```

**Expected result: zero hits.** InferenceApp currently does not reference `InferenceProvider` or `IsRocmSessionActive` anywhere. No renames needed.

If future code does reference these, the mapping is:

| Before (old) | After (new) |
|---|---|
| `InferenceProvider.Rocm` | `InferenceProvider.MIGraphX` |
| `InferenceFallbackReason.HsaRuntimeUnloadable` | Same (unchanged) |
| `InferenceFallbackReason.RocmNotAvailable` | Removed — see new granular reasons below |

### New fallback reasons

The library now provides granular fallback reasons on Linux:

| Reason | Meaning | Install command |
|---|---|---|
| `HipRuntimeMissing` | `libamdhip64.so.7` not found | `sudo apt install hip-runtime-amd` |
| `MIGraphXRuntimeMissing` | `libmigraphx_c.so.3` not found | `sudo apt install migraphx` |
| `KfdDeviceMissing` | `/dev/kfd` not found (no AMD GPU) | Hardware issue — no fix |
| `HsaRuntimeUnloadable` | `libhsa-runtime64.so` not found | `sudo apt install libhsa-runtime64` |

---

## Step 4 — Adopt the Diagnostics API (Recommended)

The new `InferenceEngine.Core.Diagnostics` namespace provides an `EngineInfo` property on every engine after `LoadAsync()`. This is optional but highly recommended for showing GPU status to users.

### 4a — Show provider status in the UI

Add a status indicator to `MainViewModel` or a new diagnostics panel:

**`ViewModels/MainViewModel.cs`** — add after `LoadImageAsync`:

```csharp
using InferenceEngine.Core;
using InferenceEngine.Core.Diagnostics;

// In MainViewModel constructor or initialization:
private string _providerStatus = "Not loaded";

// Call after engine is created (e.g. in DetectAsync after first PredictAsync)
public void UpdateProviderStatus(BaseInferenceEngine<byte[], List<OnnxInferenceEngine.PredictionResult>> engine)
{
    var info = engine.EngineInfo;
    ProviderStatus = info.Provider switch
    {
        InferenceProvider.MIGraphX => $"GPU (MIGraphX) — device {info.DeviceId}",
        InferenceProvider.DirectML => $"GPU (DirectML) — device {info.DeviceId}",
        InferenceProvider.CoreML => "GPU (CoreML)",
        InferenceProvider.Cpu when info.IsFallback => $"CPU (fallback: {info.FallbackReason})",
        InferenceProvider.Cpu => "CPU",
        _ => "Unknown",
    };
}

[ObservableProperty] private string providerStatus = "Not loaded";
```

### 4b — Use the OnSessionInitialized callback

The `InferenceEngineOptions.OnSessionInitialized` callback fires once after the session is ready, including fallback paths. This is the cleanest way to surface diagnostics without polling.

**`App.axaml.cs` — `ConfigureServices()`** — update the `IInferenceEngine` registration:

```csharp
services.AddSingleton<IInferenceEngine>(sp =>
{
    var options = new InferenceEngineOptions
    {
        UseGpuAcceleration = true,
        WarmupOnLoad = true,
        OnSessionInitialized = info =>
        {
            // This runs on a thread-pool thread, marshal to UI if needed.
            Console.WriteLine(
                $"[InferenceApp] Engine loaded: {info.Provider} " +
                $"(fallback={info.IsFallback}, reason={info.FallbackReason})");

            foreach (var check in info.Checks)
                Console.WriteLine($"  [{check.Order}] {check.Name,-30} {check.Result} {check.Detail ?? ""}");
        },
    };

    var faceAsset = Path.Combine(AppContext.BaseDirectory, "Assets", "yolov8n-face.onnx");
    if (File.Exists(faceAsset))
        options.ModelPath = faceAsset;

    var logger = NullLogger<OnnxInferenceEngine>.Instance;
    return new LazyOnnxInferenceEngine(options, logger)
    {
        Labels = ["face"],
        SingleClassFaceMode = false,
    };
});
```

### 4c — Show install instructions on fallback

When `EngineInfo.IsFallback` is true, the `Checks` trail contains actionable detail. Example for the HIP runtime check:

```csharp
OnSessionInitialized = info =>
{
    if (!info.IsFallback) return;

    var hipCheck = info.Checks.FirstOrDefault(c => c.Name == "hip.dlopen");
    if (hipCheck?.Result == CheckResult.Failed)
    {
        // Show in UI: "GPU acceleration unavailable — install hip-runtime-amd"
        // hipCheck.Detail contains the full install command.
    }
},
```

### 4d — EngineInfo properties reference

```csharp
public sealed record InferenceEngineInfo
{
    public InferenceProvider Provider { get; init; }         // MIGraphX, DirectML, CoreML, Cpu, Unknown
    public HostPlatform Platform { get; init; }              // LinuxX64, WinX64, OSXArm64, etc.
    public int DeviceId { get; init; }                       // GPU device index (default 0)
    public bool GpuAccelerationRequested { get; init; }     // What the user asked for
    public bool IsFallback { get; init; }                    // True if fell back to CPU
    public InferenceFallbackReason FallbackReason { get; init; } // Granular reason
    public IReadOnlyDictionary<string, string> ProviderOptions { get; init; }
    public string OnnxRuntimeVersion { get; init; }          // Reserved (always empty in v1)
    public DateTimeOffset SessionCreatedAt { get; init; }
    public IReadOnlyList<ProviderDiagnosticCheck> Checks { get; init; } // Ordered trail
}
```

---

## Step 5 — Update Tests

### `Tests/Tests.csproj`

Already uses `<ProjectReference>` to `InferenceApp.UI.csproj` and `<PackageReference>` to `InferenceEngine.Core`. Update the version:

```xml
<PackageReference Include="InferenceEngine.Core" Version="1.24.1-dev.*" />
```

### No test code changes needed

The test files (`OnnxInferenceEngineTests.cs`, `LazyOnnxInferenceEngineTests.cs`, `ArcFaceRecognitionEngineTests.cs`) all use the public API through `InferenceEngineOptions`, `OnnxInferenceEngine`, `LazyOnnxInferenceEngine`, and `ArcFaceRecognitionEngine`. None reference `InferenceProvider` or provider-specific enums.

### Optional: add a diagnostics test

```csharp
using InferenceEngine.Core;
using InferenceEngine.Core.Diagnostics;

[Fact]
public async Task OnnxInferenceEngine_LoadAsync_PopulatesEngineInfo()
{
    var options = new InferenceEngineOptions { ModelPath = _modelPath };
    using var engine = new OnnxInferenceEngine(options, NullLogger<OnnxInferenceEngine>.Instance);

    Assert.Equal(InferenceEngineInfo.Empty, engine.EngineInfo);

    await engine.LoadAsync();

    var info = engine.EngineInfo;
    Assert.NotEqual(InferenceProvider.Unknown, info.Provider);
    Assert.NotEmpty(info.Checks);
    Assert.True(info.Checks[0].Order == 0);
    Assert.True(info.Checks[^1].Order == info.Checks.Count - 1);
}
```

---

## Step 6 — CI/CD Updates

### `.github/workflows/ci.yml`

No changes needed. The existing workflow already passes `GITHUB_TOKEN` as `NUGET_AUTH_TOKEN`:

```yaml
- name: Restore dependencies
  run: dotnet restore
  env:
    GITHUB_TOKEN: ${{ secrets.GH_PACKAGES_TOKEN }}  # Already present
```

The `GH_PACKAGES_TOKEN` secret must have `read:packages` scope for the `intel-agency` org feed.

### `.github/workflows/publish.yml`

No changes needed. The publish workflow builds platform-specific RIDs (`linux-x64`, `win-x64`, `osx-x64`) which matches the library's architecture.

### GitHub Actions runner note

GitHub-hosted `ubuntu-latest` runners **do not have AMD GPUs**, so:

- The MIGraphX EP will never activate on CI
- Tests will fall back to CPU (expected and correct)
- The `EngineInfo.Provider` will be `Cpu` with `FallbackReason = KfdDeviceMissing`
- No GPU-specific tests will run — this is normal

---

## Step 7 — Runtime Environment Setup (Linux AMD GPU)

For local development on the Debian/AMD workstation where `/dev/kfd` is present:

### Current state

```text
libhsa-runtime64.so      FOUND   (system package: libamdhip64-5 5.7.1)
libamdhip64.so.5         FOUND   (Debian package — WRONG SONAME)
libamdhip64.so.7         MISSING (requires AMD ROCm repository)
libmigraphx_c.so.3       MISSING (requires AMD ROCm repository)
```

### Install steps

```bash
# 1. Add AMD ROCm apt repository (one-time)
#    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/native/install.html

# 2. Install runtime libraries (NOT the full ROCm stack)
sudo apt update
sudo apt install hip-runtime-amd migraphx

# 3. Verify
ldconfig -p | grep -E 'amdhip64\.so\.7|migraphx_c\.so\.3'
```

### What NOT to install

Avoid the meta-package `rocm` or `rocm-libs` — these pull in development tools,
compilers, and debug symbols (several GB). Only `hip-runtime-amd` and `migraphx`
are needed for inference.

### SONAME warning

Debian 13 (trixie) ships `libamdhip64.so.5` (HIP 5.7.1). MIGraphX links against
`libamdhip64.so.7` (HIP 7.x). These are **ABI-incompatible** — you cannot symlink
`.so.5` → `.so.7`. The AMD ROCm repository must be used.

---

## Step 8 — Deployment Checklist

Before shipping InferenceApp with the updated library:

- [ ] Update `InferenceEngine.Core` package version in both `.csproj` files
- [ ] Update `Microsoft.ML.OnnxRuntime` to `1.24.1` in `InferenceApp.UI.csproj`
- [ ] Run `dotnet restore && dotnet build && dotnet test` locally
- [ ] Verify CI passes on all three platforms (ubuntu-latest, windows-latest, macos-latest)
- [ ] Verify publish workflow produces artifacts for `linux-x64`, `win-x64`, `osx-x64`
- [ ] On the AMD workstation: confirm MIGraphX activates (check `EngineInfo.Provider == MIGraphX`)
- [ ] On non-GPU machines: confirm CPU fallback works with actionable diagnostics
- [ ] (Optional) Add `OnSessionInitialized` callback to surface GPU status in UI

---

## Architecture: How Provider Selection Works

```text
BaseInferenceEngine.LoadAsync()
  │
  ├─ ConfiguredSessionOptions()
  │   │
  │   ├─ UseGpuAcceleration == false? → CPU
  │   │
  │   ├─ Linux x64 → TryRocmBranch()
  │   │   ├─ /dev/kfd exists?
  │   │   ├─ libhsa-runtime64.so loadable?
  │   │   ├─ libamdhip64.so.7 loadable?      ← NEW: HIP runtime probe
  │   │   ├─ libmigraphx_c.so.3 loadable?    ← NEW: MIGraphX runtime probe
  │   │   ├─ AppendExecutionProvider_MIGraphX() succeeds?
  │   │   └─ InferenceSession() constructs?
  │   │       ↓ (any fail)
  │   │       CPU fallback with granular reason
  │   │
  │   ├─ Windows → TryDirectMlBranch()
  │   ├─ macOS → TryCoreMlBranch()
  │   └─ Other → CPU
  │
  ├─ EngineInfo populated with full check trail
  └─ OnSessionInitialized callback fires
```

Each probe generates a `ProviderDiagnosticCheck` with `Passed`, `Failed`, or `Skipped` result.
The full trail is available via `engine.EngineInfo.Checks` after `LoadAsync()`.

---

## Diagnostic Check Names (Linux x64)

| Check Name | Category | What It Tests |
|---|---|---|
| `gpu.user-enabled` | Config | `UseGpuAcceleration == true` |
| `rid.platform` | Platform | Host is Linux x64 |
| `kfd.present` | Provider | `/dev/kfd` exists |
| `hsa.dlopen` | Provider | `libhsa-runtime64.so` loads |
| `hip.dlopen` | Provider | `libamdhip64.so.7` loads |
| `migraphx.dlopen` | Provider | `libmigraphx_c.so.3` loads |
| `migraphx.append` | Provider | `AppendExecutionProvider_MIGraphX()` succeeds |
| `migraphx.session` | Session | `new InferenceSession()` succeeds |
| `cpu.fallback-options` | Fallback | CPU session options created (fallback path only) |
| `cpu.fallback-session` | Fallback | CPU session constructed (fallback path only) |

---

## Breaking Changes Reference

| What Changed | Impact on InferenceApp |
|---|---|
| `InferenceProvider.Rocm` → `MIGraphX` | **Zero impact** — InferenceApp doesn't reference this enum |
| `IsRocmSessionActive` semantics | **Zero impact** — not referenced in InferenceApp |
| ORT version `1.26.0` → `1.24.1` | **Must downgrade** managed bindings to match native libs |
| New `InferenceFallbackReason` values | **Zero impact** — only accessible via `EngineInfo` (opt-in) |
| `EngineInfo` property added to `BaseInferenceEngine` | **Additive** — opt-in, no existing code breaks |
| `OnSessionInitialized` callback added | **Additive** — opt-in, no existing code breaks |
