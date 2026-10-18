# Design: Out-of-Process Inference Engine (`OutOfProcessBaseInferenceEngine`)

**Status:** Proposal / handoff for the InferenceEngine.Core team
**Author:** Consumer app team (context: AMD ROCm / MIGraphX bring-up)
**Date:** 2026-07-05
**Target:** `InferenceEngine.Core` (this repo)
**Related:** `docs/linux-rocm-config.md`, `docs/engine-info-api-plan-v2.md`

---

## TL;DR

MIGraphX GPU inference and an Avalonia GPU-rendered UI **cannot coexist in one process** on a standard Debian/Mesa + ROCm stack: Mesa's OpenGL stack pulls in the system `libLLVM-19`, and ROCm's `comgr` (lazily loaded by MIGraphX at compile time) carries a private LLVM-20 whose `ManagedStatic` CommandLine registry collides with it → `LLVM ERROR: inconsistency in registered CommandLine options` → `SIGABRT`. Every in-process workaround was tried and either fails or cripples the UI.

The proposed fix is a **new base class in `InferenceEngine.Core`**, `OutOfProcessBaseInferenceEngine<TInput, TOutput>`, that offers the **exact same contract** as `BaseInferenceEngine<TInput, TOutput>` but runs the ONNX session in a **child worker process** over a pipe. The host (UI) process never loads `libonnxruntime`/`comgr`; the worker process never loads Mesa/GL. Two processes, two LLVMs, zero collision — and the consumer API is unchanged.

This keeps **GPU UI rendering** *and* **GPU inference** with no compromise.

---

## 1. Context & background (why this is needed now)

This work originated from getting MIGraphX to actually load in **the consumer app** (Avalonia UI, .NET 10) on a Debian 13 / AMD Radeon RX 6700 XT (gfx1031) host. The journey, with evidence the team should not have to re-derive:

1. **EP code already shipped.** `InferenceEngine.Core` `1.1.0-dev.x` contains the full MIGraphX/HIP path: `BaseInferenceEngine.ConfiguredSessionOptions` → `TryRocmBranch()` probes `/dev/kfd` → `libhsa-runtime64.so` → `libamdhip64.so.7` → `libmigraphx_c.so.3` → `AppendExecutionProvider_MIGraphX(deviceId)` → `new InferenceSession(...)`, with CPU fallback + the `EngineInfo` diagnostic trail.
2. **Package + native-lib fixes landed.** A consumer-side build target (`ForceRocmOnnxRuntimeLibOnLinux`, now also shipped via this package's `buildTransitive/InferenceEngine.Core.targets`) ensures the ROCm-enabled `libonnxruntime.so` (which exports `OrtSessionOptionsAppendExecutionProvider_MIGraphX`) wins over the vanilla CPU one. See commit history.
3. **Host ROCm 7.1 installed** (`hip-runtime-amd 7.1`, `migraphx 2.14`). `hip.dlopen` + `migraphx.dlopen` checks PASS.
4. **Result:** in a **GL-free** process, MIGraphX loads and runs inference correctly. The consumer app's `--diag inference [--image <face.jpg>]` CLI prints `Provider : MIGraphX (device 0)`, all checks PASS, `[diag] inference ok detections=1`, exit 0. **MIGraphX works.**

The remaining problem is **only** about running MIGraphX *inside the full Avalonia UI process*.

---

## 2. Use case

**The consumer app** is an Avalonia (.NET 10) desktop app. It uses:
- **Avalonia GPU rendering** (OpenGL/Skia on X11) for its UI, and
- **MIGraphX inference** (detection + recognition) on the AMD GPU.

Both must run in the same application. Today the engine is constructed in-process via DI (`OnnxInferenceEngine : BaseInferenceEngine<byte[], List<PredictionResult>>`). On a ROCm host this combination aborts at startup (see §3). The product requirement is: **GPU-rendered UI + GPU inference, no software-rendering fallback.**

This is not consumer-app-specific: any consumer that renders with a GL/Mesa stack (i.e., most GUI apps) and wants MIGraphX on AMD/ROCm will hit the same wall. Putting the solution in `InferenceEngine.Core` benefits all consumers and matches the library's stated role — **platform detection and GPU acceleration are hidden inside the library; consumers just subclass/use the engine.**

---

## 3. The blocker: in-process LLVM collision

### Symptom
Launching the full app (Avalonia UI) on the ROCm host aborts during the first MIGraphX program compile:
```
mesa: CommandLine Error: Option 'allow-incomplete-ir' registered more than once!
LLVM ERROR: inconsistency in registered CommandLine options
            → SIGABRT (exit 134)
```

### Mechanism (verified)
- Avalonia's OpenGL init causes the .NET runtime to `dlopen("libGL.so.1")` → Mesa `libGLX_mesa` → **`libgallium-25.0.7-2.so`** → **`/usr/lib/x86_64-linux-gnu/libLLVM.so.19.1`** (system LLVM 19). Captured via `LD_DEBUG=files`:
  ```
  file=libGL.so.1; dynamically loaded by libcoreclr.so
  ...
  file=libLLVM.so.19.1; needed by /lib/x86_64-linux-gnu/libgallium-25.0.7-2.so
  ```
- MIGraphX lazily `dlopen`s ROCm's **`/opt/rocm/lib/libamd_comgr.so`** at compile time (it is **not** a `NEEDED` of `libmigraphx_c.so.3`). `libamd_comgr.so` **statically embeds LLVM-20** (no `NEEDED libLLVM*`; the string `allow-incomplete-ir` is present inside it).
- When comgr initializes, its LLVM-20 `cl::opt` for `allow-incomplete-ir` is registered into a `ManagedStatic` CommandLine registry that collides with the already-loaded system `libLLVM-19`'s registry → LLVM detects the duplicate → `report_fatal_error` → `SIGABRT`.

**Both LLVMs are required** (Mesa's for the GL renderer, comgr's for MIGraphX compilation), and neither can be removed without losing GPU rendering or GPU inference.

### Proof the conflict is process-scoped, not engine-scoped
The `--diag inference` path (same `OnnxInferenceEngine`, same MIGraphX EP, same model) runs in the **same executable but exits before Avalonia starts**, so Mesa/`libLLVM-19` is never loaded → comgr's LLVM is the only one → **no crash, MIGraphX inference succeeds**. This is the existence proof for the out-of-process approach: move the session into a process with no GL, and the collision disappears.

### In-process remedies attempted (all rejected/failed)
| Attempt | Result |
|---|---|
| Avalonia `X11RenderingMode.Software` (no GL) | ✅ works — **rejected**: defeats GPU UI rendering |
| `MESA_LOADER_DRIVER_OVERRIDE=zink` (GL-over-Vulkan) | ❌ `libgallium` links `libLLVM-19` for *every* gallium driver incl. Zink |
| `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libLLVM.so.19.1` | ❌ collision persists |
| `LIBGL_ALWAYS_SOFTWARE=1` | ❌ gallium still loads `libLLVM-19` |
| MIGraphX persistent compile-cache (skip comgr) | ❌ MIGraphX 2.14 exposes no on-disk compile cache (no cache env/dir) |

**Conclusion:** on the current Debian/Mesa + ROCm-7.1 stack, in-process GL rendering + MIGraphX is a dead end. The clean fix is to put the ONNX session (and therefore comgr) in a separate process.

> Environment of record: Debian GNU/Linux 13 (trixie); AMD Radeon RX 6700 XT (gfx1031, requires `HSA_OVERRIDE_GFX_VERSION=10.3.0`); ROCm 7.1 (`hip-runtime-amd 7.1`, `migraphx 2.14`, `comgr 3.0.0.70100`); Mesa 25.0.7 gallium (links `libLLVM-19`); Avalonia 12.0.4; `Microsoft.ML.OnnxRuntime` 1.24.1 (custom ROCm native build 1.24.1.36).

---

## 4. Proposed solution

A new base class in `InferenceEngine.Core` that is a **transparent proxy** with the same contract as `BaseInferenceEngine`, delegating to a worker process.

### 4.1 Host side — proxy base class

```csharp
namespace InferenceEngine.Core;

// Same contract as BaseInferenceEngine<TInput,TOutput>: LoadAsync, PredictAsync,
// EngineInfo, IsRocmSessionActive, FallbackReason, etc. Consumers (or a thin
// consumer wrapper) use it exactly like an in-process engine.
public abstract class OutOfProcessBaseInferenceEngine<TInput, TOutput>
    : BaseInferenceEngine<TInput, TOutput>
{
    protected OutOfProcessBaseInferenceEngine(
        OutOfProcessEngineOptions options,
        ILogger logger);

    // Overrides do NOT create a session in-process (the host must never dlopen
    // libonnxruntime/comgr). They marshal to the worker. EngineInfo /
    // IsRocmSessionActive are populated from the worker's LoadResponse, so the
    // EngineInfo diagnostics API behaves identically to the in-process engine.
    // Signatures match the existing contract (IInferenceEngine<TInput,TOutput> /
    // BaseInferenceEngine.LoadAsync()/PredictAsync(TInput) — no CancellationToken today).
    public override Task LoadAsync();
    public override Task<TOutput> PredictAsync(TInput input);
}

public sealed class OutOfProcessEngineOptions
{
    /// <summary>Absolute path to the consumer's worker executable.</summary>
    public required string WorkerExecutablePath { get; init; }

    /// <summary>Identifies which engine the worker should construct (consumer-defined).</summary>
    public string EngineId { get; init; } = "";

    /// <summary>Env for the worker only (e.g. HSA_OVERRIDE_GFX_VERSION=10.3.0). Never set on the host.</summary>
    public IReadOnlyDictionary<string, string> WorkerEnvironment { get; init; } = new Dictionary<string, string>();

    public InferenceEngineOptions EngineOptions { get; init; } = new();

    public TimeSpan StartupTimeout { get; init; } = TimeSpan.FromMinutes(3); // first MIGraphX compile is slow
    public TimeSpan InferenceTimeout  { get; init; } = TimeSpan.FromSeconds(60);
}
```

**Critical invariant:** the proxy's `LoadAsync`/`PredictAsync` **must not** touch any P/Invoke into `onnxruntime`. Because P/Invoke resolution is lazy, merely loading `InferenceEngine.Core.dll` in the host is safe; the host process therefore stays free of `libonnxruntime`/`comgr` and only carries Mesa/`libLLVM-19` (which is harmless on its own).

**Contract alignment (grounded in the current code).** `IInferenceEngine<TInput, TOutput>` **already exists** and is the documented contract; `BaseInferenceEngine<TInput, TOutput>` implements it, with `LoadAsync`/`PredictAsync` declared `virtual`. `BaseStreamingInferenceEngine<TInput, TOutput, TStep>` is the established **sibling pattern**: it extends `BaseInferenceEngine`, keeps the base's in-process session, and overrides `PredictAsync` to route through a different execution path (the decode loop), sealing the one-shot `PreProcessWithContext`/`PostProcessWithContext` hooks. `OutOfProcessBaseInferenceEngine` follows the same sibling idea, with one crucial difference: it must **not** create an in-process session at all.

**Design fork forced by the current code:** `EngineInfo` and `IsRocmSessionActive` on `BaseInferenceEngine` have **`private set`** — a subclass cannot assign them. The proxy must mirror these from the worker's `LoadResponse`. Two options:
- **(1) Extend `BaseInferenceEngine`** and add a narrow protected seam (e.g. `protected void ApplyRemoteEngineInfo(InferenceEngineInfo info, bool isRocmActive)`) so the proxy can populate them. Reuses the full `BaseInferenceEngine` surface (`InputMetadata`, `Dispose`, etc.) and matches `BaseStreamingInferenceEngine`'s "extend + override" style. Cost: a small base-class change (a guarded protected setter).
- **(2) Implement `IInferenceEngine<TInput, TOutput>` directly**, with the proxy's own `EngineInfo`/`IsRocmSessionActive` properties. Cleanest isolation (no base machinery, no `_session` field), no base change — but diverges from "everything derives from `BaseInferenceEngine`," so DI/consumers that depend on `BaseInferenceEngine`-specific members would need to depend on the interface instead.

**Recommendation:** **(1)** with a narrow protected seam, for consistency with streaming and to keep the diagnostics surface transparent. This is the team's call (§5). A future `OutOfProcessStreamingInferenceEngine` could compose streaming + OOP cleanly if both honor the `IInferenceEngine` interface.

### 4.2 Worker side — reusable host loop (library)

The library can't instantiate a consumer's engine subclass (`OnnxInferenceEngine` lives in the consumer). So the library provides the **message loop**, and the consumer provides the **engine factory** + process entry point:

```csharp
namespace InferenceEngine.Core.Worker;

public static class InferenceWorkerHost
{
    // The consumer's worker Program calls this with a factory that builds ITS engine.
    // The worker is just a normal BaseInferenceEngine — exactly like the already-working
    // `--diag inference` CLI: it loads the model, runs EP selection (TryRocmBranch →
    // MIGraphX), and serves requests. Because it never loads Avalonia/GL, comgr's LLVM is
    // the only LLVM in the process → no collision.
    public static Task<int> RunAsync<TInput, TOutput>(
        Func<InferenceEngineOptions, BaseInferenceEngine<TInput, TOutput>> engineFactory,
        IWorkerSerializer<TInput, TOutput>? serializer = null,
        CancellationToken ct = default);
}
```

The loop is a strict request/response state machine over stdin/stdout (stderr passes through to the host's logs). One request is in flight at a time. Pseudocode:

```csharp
BaseInferenceEngine<TInput, TOutput>? engine = null;
while (await protocol.ReadAsync(ct) is { } msg)
{
    switch (msg)
    {
        case LoadRequest r:
            engine = engineFactory(r.Options);
            try {
                await engine.LoadAsync();                                   // real EP selection happens here
                await protocol.WriteAsync(new LoadResponse(engine.EngineInfo, engine.IsRocmSessionActive, Error: null));
            }
            catch (Exception ex) {
                await protocol.WriteAsync(new LoadResponse(engine.EngineInfo, IsRocmSessionActive: false, Error: ex.ToString()));
            }
            break;

        case PredictRequest p when engine is not null:
            try {
                var output = await engine.PredictAsync(serializer.DeserializeInput(p.Input));
                await protocol.WriteAsync(new PredictResponse(serializer.SerializeOutput(output), Error: null));
            }
            catch (Exception ex) {
                await protocol.WriteAsync(new PredictResponse(Output: null, Error: ex.ToString()));
            }
            break;

        case Shutdown: return 0;
    }
}
```

On `Load`, the worker's real engine runs the full probe chain and EP selection (`TryRocmBranch` → MIGraphX) and emits its `EngineInfo` (provider, `IsRocmSessionActive`, fallback reason, ordered `Checks`) back; the host proxy mirrors that onto itself (see the design fork in §4.1). On `Predict`, it runs the model and returns the serialized `TOutput`.

### 4.3 IPC protocol (length-prefixed JSON over stdin/stdout)

**Framing:** every message is `<4-byte little-endian length><UTF-8 JSON>`. A `kind` discriminator routes each message; one request in flight at a time, responses matched in order.

```jsonc
// host → worker
{ "kind": "load", "engineId": "onnx-yolo-face", "options": { "modelPath": "...", "useGpuAcceleration": true } }

// worker → host  (EngineInfo is the same shape the in-process engine produces today)
{ "kind": "loaded", "engineInfo": {
      "platform": "LinuxX64", "provider": "MIGraphX", "deviceId": 0,
      "gpuAccelerationRequested": true, "isFallback": false, "fallbackReason": null,
      "checks": [ { "order": 0, "name": "gpu.user-enabled", "result": "Passed", "detail": "..." } ]
  }, "isRocmSessionActive": true, "error": null }

// host → worker  (TInput via IWorkerSerializer — byte[] as base64 here)
{ "kind": "predict", "input": "<base64 image bytes>" }

// worker → host  (TOutput via IWorkerSerializer — List<PredictionResult> as JSON here)
{ "kind": "result", "output": [ { "label": "face", "confidence": 0.97, "box": { "x": 12, "y": 34, "w": 56, "h": 78 } } ], "error": null }

// host → worker (graceful)
{ "kind": "shutdown" }
```

**Serialization seam:** because `TInput`/`TOutput` cross the boundary, introduce `IWorkerSerializer<TInput, TOutput>` with a default `System.Text.Json` implementation. For `OnnxInferenceEngine` (`byte[]` → `List<PredictionResult>`) this is trivial (base64 input, JSON detections). The generic base must not assume arbitrary `TInput`/`TOutput` are serializable — constrain or require the serializer.

### 4.4 Consumer side (the consumer app) — worker entry point + opt-in wiring

Mirrors the existing `--diag inference` headless path:

```csharp
// consumer-ui/Program.cs
if (args.Length > 1 && args[0] == "--worker" && args[1] == "inference")
    return InferenceWorkerHost.RunAsync<byte[], List<PredictionResult>>(
        opts => new OnnxInferenceEngine(opts, logger)).GetAwaiter().GetResult();
```

The consumer decides **when** to use OOP (opt-in). The library cannot reliably detect "the host has a GL context loaded," so the host chooses — e.g. the consumer app uses the OOP proxy on ROCm hosts (`libamdhip64.so.7` present) and the in-process engine elsewhere:

```csharp
IInferenceEngine engine = OperatingSystem.IsLinux() && NativeLibrary.TryLoad("libamdhip64.so.7", out _)
    ? new OutOfProcessOnnxInferenceEngine(workerOpts, logger)   // thin consumer wrapper over the proxy
    : new OnnxInferenceEngine(opts, logger);
```

### 4.5 Why this preserves the contract
- **Diagnostics:** `EngineInfo` (provider, fallback reason, ordered `Checks`, `IsRocmSessionActive`) is produced by the worker's real engine and mirrored onto the proxy. Existing consumers of the EngineInfo API (e.g. the consumer app's `EngineDiagnosticsViewModel` via `OnSessionInitialized`) keep working unchanged.
- **EP selection:** `TryRocmBranch()` etc. run in the worker (a normal `BaseInferenceEngine`). The host never runs EP selection.
- **Error semantics:** worker exceptions are surfaced as the same exception types / `EngineInfo` fallback states the in-process engine would produce.

---

## 5. Decisions / open questions for the team

1. **Extend `BaseInferenceEngine` (with a protected seam) vs implement `IInferenceEngine` directly.** `IInferenceEngine<TInput, TOutput>` **already exists** — the open question is what the proxy derives from. `BaseStreamingInferenceEngine` sets the precedent of "extend `BaseInferenceEngine`, override execution," but — unlike streaming — the OOP proxy must hold **no** in-process session and must mirror `EngineInfo`/`IsRocmSessionActive` from the worker (currently `private set`). **Recommend** extend-`BaseInferenceEngine` + a narrow protected seam (`ApplyRemoteEngineInfo`) for consistency and to preserve the diagnostics surface transparently; the team may instead prefer implement-`IInferenceEngine`-directly for stronger isolation (no `_session` field, no base change, but consumers depend on the interface, not `BaseInferenceEngine`).
2. **Serialization constraint on `TInput`/`TOutput`.** Default JSON serializer; require `IWorkerSerializer` for non-trivially-serializable types. Decide whether to constrain the generic params.
3. **One worker per engine, or a multi-engine worker.** The consumer app needs detection + recognition. Recommend **one worker per engine** (simplest lifecycle, isolated MIGraphX compile) unless profiling favors a shared worker.
4. **Worker discovery.** `WorkerExecutablePath` is consumer-supplied (the library can't know the consumer's exe). Provide a sensible default convention (e.g. same exe + `--worker <id>`) and let the consumer override.
5. **Auto vs opt-in.** Keep opt-in (host decides). Optionally add a helper `OutOfProcessEngineOptions.ForCurrentProcess()` that detects ROCm capability to reduce consumer boilerplate — but never auto-launch a worker without consumer consent.
6. **Lifecycle/resilience.** Worker warm-up state, crash detection + supervised restart, startup/inference timeouts, graceful shutdown on host exit, and a loading/progress surface for the ~85s first MIGraphX compile.
7. **Cross-platform scope.** OOP is only needed on Linux/ROCm (the LLVM collision is Mesa+comgr specific). The feature can be built generically; the *use* is Linux/ROCm. No Windows/macOS behavior change expected.

---

## 6. Caveats

- **IPC cost per inference** is negligible for image/single-shot inference. For high-FPS video, design the worker to stay hot and accept a frame stream rather than one process-round-trip per frame.
- **First-compile latency** (~85s on a 6700 XT) is incurred in the worker; the host should expose a loading state and not block the UI thread.
- **`HSA_OVERRIDE_GFX_VERSION=10.3.0`** (gfx1031 / RX 6700 XT) must be set **on the worker's environment only** (`WorkerEnvironment`), not the host's.
- The worker must load the ROCm `libonnxruntime.so` (the `ForceRocmOnnxRuntimeLibOnLinux`/`buildTransitive` target already guarantees this for consumers).

---

## 7. Acceptance criteria

- [ ] `OutOfProcessBaseInferenceEngine<TInput,TOutput>` exists in `InferenceEngine.Core`, derives from `BaseInferenceEngine<TInput,TOutput>`, and exposes `LoadAsync`/`PredictAsync`/`EngineInfo`/`IsRocmSessionActive`/`FallbackReason` with identical semantics.
- [ ] `InferenceWorkerHost.RunAsync` runs a consumer-provided engine over the documented IPC protocol.
- [ ] Host process loads **no** `onnxruntime`/`comgr` native lib (verify with `LD_DEBUG=files` / `lsof`); worker process loads **no** `libGL`/`libLLVM-19`.
- [ ] Repro: on the Debian/ROCm/gfx1031 host, the consumer app launches with **GPU rendering**, runs MIGraphX inference, captures output, exit 0 — **no** `LLVM ERROR`.
- [ ] `EngineInfo` reported by the proxy matches the worker's (provider=`MIGraphX`, `IsRocmSessionActive=true`, full check trail) — diagnostics UI unchanged.
- [ ] Existing in-process engine behavior and all current tests unchanged; OOP is purely additive and opt-in.

---

## 8. References / repro

- **Migration context:** `docs/linux-rocm-config.md`.
- **EngineInfo API (mirrored over IPC):** `docs/engine-info-api-plan-v2.md`.
- **Repro the working path (GL-free):**
  ```bash
  HSA_OVERRIDE_GFX_VERSION=10.3.0 dotnet run --project consumer-ui -c Release -- --diag inference --image <face.jpg>
  # → Provider : MIGraphX, all checks PASS, detections=N, exit 0
  ```
- **Repro the crash (full UI, in-process):** launch the consumer app UI on the ROCm host → `LLVM ERROR: … registered more than once` → SIGABRT.
- **Confirm the dlopen chain:** run the UI under `LD_DEBUG=files` and grep for `libGL.so.1 … libcoreclr.so` and `libLLVM.so.19.1 … libgallium`.
- Consumer-app artifacts: `--diag inference` CLI (`consumer-ui/Diagnostics/DiagInferenceCli.cs`), the `ForceSoftwareRendering` opt-in flag (`AppSettings` / `FORCE_SOFTWARE_RENDERING`), and the launch config fix in `.vscode/launch.json`.
