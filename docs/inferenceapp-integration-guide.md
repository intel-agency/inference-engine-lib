# Integrating InferenceEngine.Core into InferenceApp

**Target repo:** `nam20485/InferenceApp`  
**Package:** `InferenceEngine.Core` published from `intel-agency/inference-engine-lib`  
**Feed:** GitHub Packages — `https://nuget.pkg.github.com/intel-agency/index.json`

---

## Overview

`InferenceEngine.Core` is the canonical managed library. Reference it as a NuGet package.
Do **not** copy the source, use a git submodule, or maintain a local fork.

The full transitive dependency chain is automatic:

```text
InferenceEngine.Core (NuGet)
  ├── Microsoft.ML.OnnxRuntime               (all platforms — CPU baseline)
  ├── Microsoft.ML.OnnxRuntime.DirectML      (Windows only — AMD/Intel/NVIDIA GPU)
  ├── InferenceEngine.ROCm.Runtime.linux-x64 (Linux x64 only — AMD ROCm GPU)
  └── Microsoft.ML.OnnxRuntime + CoreML EP   (macOS — via base package)
```

No additional steps are required to get hardware acceleration — it is selected
automatically at runtime by `BaseInferenceEngine` based on the current OS.

---

## Step 1 — Add nuget.config

Create (or update) `nuget.config` at the **root** of the InferenceApp repository:

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" protocolVersion="3" />
    <add key="github-intel-agency" value="https://nuget.pkg.github.com/intel-agency/index.json" />
  </packageSources>
  <packageSourceCredentials>
    <!--
      Set NUGET_AUTH_TOKEN in your environment to a GitHub PAT with read:packages scope.
      In CI, pass GITHUB_TOKEN as NUGET_AUTH_TOKEN (see Step 4).
      DO NOT commit real credentials here.
    -->
    <github-intel-agency>
      <add key="Username" value="x-access-token" />
      <add key="ClearTextPassword" value="%NUGET_AUTH_TOKEN%" />
    </github-intel-agency>
  </packageSourceCredentials>
</configuration>
```

---

## Step 2 — Remove the Existing Local Copy

InferenceApp currently contains a copy of the `InferenceEngine.Core` source under
`inference-engine-lib/`. This must be removed — it is replaced by the NuGet package.

```bash
# From InferenceApp repo root
git rm -r inference-engine-lib/
```

Remove any solution or project references that pointed at the local copy (`.slnx`,
`Directory.Build.props`, etc.).

---

## Step 3 — Add PackageReference

In each `.csproj` that previously referenced the local `InferenceEngine.Core` project,
replace the `ProjectReference` with a `PackageReference`:

**Before:**

```xml
<ProjectReference Include="..\inference-engine-lib\InferenceEngine.Core.csproj" />
```

**After:**

```xml
<PackageReference Include="InferenceEngine.Core" Version="1.0.0-dev.*" />
```

Use `*` wildcard to float to the latest dev build, or pin to a specific version found
on the [GitHub Packages feed](https://github.com/orgs/intel-agency/packages?repo_name=inference-engine-lib).

**Version scheme:**

| Branch in inference-engine-lib | Version format | Example |
| --- | --- | --- |
| `development` | `{PREFIX}-dev.{N}` | `1.0.0-dev.42` |
| `staging` | `{PREFIX}-rc.{N}` | `1.0.0-rc.7` |
| `release` | `{PREFIX}.{N}` | `1.0.0.71` |

---

## Step 4 — Local Development Credentials

Before running `dotnet restore` locally, set the `NUGET_AUTH_TOKEN` environment
variable to a GitHub PAT with `read:packages` scope:

**PowerShell (Windows):**

```powershell
$env:NUGET_AUTH_TOKEN = "ghp_yourpat..."
dotnet restore
```

**Bash (Linux/macOS/WSL2):**

```bash
export NUGET_AUTH_TOKEN="ghp_yourpat..."
dotnet restore
```

The PAT only needs `read:packages` — no write access required for consumers.

---

## Step 5 — CI Configuration

Pass `GITHUB_TOKEN` as `NUGET_AUTH_TOKEN` during the restore step in any GitHub
Actions workflow:

```yaml
- name: Restore dependencies
  env:
    NUGET_AUTH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: dotnet restore
```

> `GITHUB_TOKEN` has `read:packages` scope by default for workflows in repos owned
> by the same organization (`intel-agency` / `nam20485`). If the workflow is in a
> different org, use a PAT stored as a repository secret instead.

---

## Step 6 — Verify

```bash
dotnet restore
dotnet build --configuration Release
dotnet test
```

On Linux x64 with an AMD GPU, you can additionally verify ROCm is active:

```bash
dotnet test --filter "FullyQualifiedName~FaceDetectionEngineTests"
# PredictAsync_OnLinuxX64WithAmdGpu_ShouldUseRocmProvider should PASS (not Skip)
```

---

## Using InferenceEngine.Core in Application Code

Reference the public API through `IInferenceEngine<TInput, TOutput>`:

```csharp
using InferenceEngine.Core;
using InferenceEngine.Examples;
using Microsoft.Extensions.Logging;

// 1. Create an engine (FaceDetectionEngine is a pre-built example)
var engine = new FaceDetectionEngine(
    new InferenceEngineOptions
    {
        UseGpuAcceleration = true,   // DirectML on Windows, ROCm on Linux, CoreML on macOS
        WarmupOnLoad = true,
    },
    loggerFactory.CreateLogger<FaceDetectionEngine>());

// 2. Load (downloads model on first run, cached afterwards)
await engine.LoadAsync();

// 3. Predict
byte[] imageBytes = File.ReadAllBytes("photo.jpg");
var detections = await engine.PredictAsync(imageBytes);
```

For a custom prediction type, inherit from `BaseInferenceEngine<TInput, TOutput>` and
implement two methods:

```csharp
public class MyEngine : BaseInferenceEngine<MyInput, MyOutput>
{
    protected override string ModelName => "my-model.onnx";

    protected override (List<NamedOnnxValue> InputTensors, object? Context)
        PreProcessWithContext(MyInput input) { ... }

    protected override MyOutput
        PostProcessWithContext(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output,
                               object? context) { ... }
}
```

See [AGENTS.md](../AGENTS.md) for the full custom engine checklist.

---

## Diagnostics

`BaseInferenceEngine<TInput, TOutput>` exposes an `EngineInfo` property of type
`InferenceEngineInfo` (in `InferenceEngine.Core.Diagnostics`) after `LoadAsync`
completes. This immutable snapshot contains the resolved execution provider,
platform, and an ordered diagnostic trail explaining *why* a particular provider
was selected — or why the engine fell back to CPU.

### Reading EngineInfo

```csharp
using InferenceEngine.Core;
using InferenceEngine.Core.Diagnostics;

await engine.LoadAsync();

InferenceEngineInfo info = engine.EngineInfo;
Console.WriteLine($"Provider:  {info.Provider}");
Console.WriteLine($"Platform:  {info.Platform}");
Console.WriteLine($"Fallback:  {info.IsFallback} (reason: {info.FallbackReason})");

foreach (var check in info.Checks)
{
    Console.WriteLine($"  [{check.Order}] {check.Name,-25} {check.Result,-8} {check.Detail ?? ""}");
}
```

**Example output on Linux x64 with ROCm:**

```
Provider:  Rocm
Platform:  LinuxX64
Fallback:  False (reason: None)
  [0] gpu.user-enabled            Passed
  [1] rid.platform                Passed
  [2] kfd.present                 Passed
  [3] hsa.dlopen                  Passed
  [4] rocm.append                 Passed
  [5] rocm.session                Passed
```

**Example output on Linux x64 without ROCm (no /dev/kfd):**

```
Provider:  Cpu
Platform:  LinuxX64
Fallback:  True (reason: KfdDeviceMissing)
  [0] gpu.user-enabled            Passed
  [1] rid.platform                Passed
  [2] kfd.present                 Failed    /dev/kfd not found
  [3] hsa.dlopen                  Skipped   precondition kfd.present failed
  [4] rocm.append                 Skipped   precondition kfd.present failed
  [5] rocm.session                Skipped   precondition kfd.present failed
  [6] cpu.session                 Passed
```

### OnSessionInitialized Callback

`InferenceEngineOptions` has an optional `OnSessionInitialized` callback that
fires once after the session is created (including fallback paths):

```csharp
var engine = new FaceDetectionEngine(
    new InferenceEngineOptions
    {
        UseGpuAcceleration = true,
        OnSessionInitialized = info =>
        {
            Console.WriteLine($"Engine loaded: {info.Provider} (fallback={info.IsFallback})");
        },
    },
    logger);
```

---

## Package Version Pinning Strategy

| Scenario | Recommended version expression |
| --- | --- |
| Active development against latest | `1.0.0-dev.*` |
| Stabilizing for a milestone | `1.0.0-rc.*` |
| Production / published release | `1.0.0.*` (floating patch) or exact |

To see available versions:

```bash
# Requires NUGET_AUTH_TOKEN to be set
dotnet package search InferenceEngine.Core --source github-intel-agency --prerelease
```

---

## Troubleshooting

| Error | Cause | Fix |
| --- | --- | --- |
| `Unable to load the service index for source https://nuget.pkg.github.com/intel-agency/index.json` | `NUGET_AUTH_TOKEN` not set or invalid | Set env var to a PAT with `read:packages` |
| `Package 'InferenceEngine.Core X.Y.Z' not found` | Version published from `development` branch not yet available | Check [GitHub Packages](https://github.com/orgs/intel-agency/packages?repo_name=inference-engine-lib); push to `development` triggers publish |
| `Could not load file or assembly 'InferenceEngine.ROCm.Runtime.linux-x64'` | Building without `--runtime linux-x64` | Add `-r linux-x64` or `<RuntimeIdentifier>linux-x64</RuntimeIdentifier>` in `.csproj` |
| ROCm test skips on Linux runner | `/dev/kfd` not present — no AMD GPU or session issue | See [rocm-wsl2-runner-setup.md](rocm-wsl2-runner-setup.md) |
