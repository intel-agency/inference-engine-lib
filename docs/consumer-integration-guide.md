# Consumer Integration Guide — Choosing an Inference API

> **Audience:** developers building a consuming application on top of `InferenceEngine.Core`. This
> guide is self-contained and tells you which of the three public API surfaces to use and how to wire
> each one up.

## Overview & package

`InferenceEngine.Core` is one library that exposes **three additive API surfaces** for running ONNX
models with automatic hardware acceleration (MIGraphX on Linux/ROCm, DirectML on Windows, CoreML on
macOS, with graceful CPU fallback). All three share the same lifecycle (`LoadAsync` → `PredictAsync`
→ `Dispose`), the same diagnostics (`EngineInfo` + ordered `Checks`), and the same model-download
plumbing. **One-shot always works; the other two are pure additions.** On top of those surfaces the library
also ships **ready-made text engines** (embeddings + text generation) so most string-in/string-out
consumers never write a subclass at all.

```xml
<PackageReference Include="InferenceEngine.Core" Version="1.1.0-dev.13" />
```

For full package-setup details (the GitHub Packages feed, `nuget.config`, credentials, CI, and
version pinning) see the root [`README.md`](../README.md) and [`nuget.config`](../nuget.config).

| Surface | Base class | Contract | Yields |
|---|---|---|---|
| **A. One-shot** | `BaseInferenceEngine<TInput,TOutput>` | `IInferenceEngine<TInput,TOutput>` | One fully-reduced result per call |
| **B. Streaming** | `BaseStreamingInferenceEngine<TInput,TOutput,TStep>` | `IStreamingInferenceEngine<TInput,TOutput,TStep>` | One model-level step at a time (`IAsyncEnumerable<TStep>`) |
| **C. Out-of-process** | `OutOfProcessBaseInferenceEngine<TInput,TOutput>` | `IInferenceEngine<TInput,TOutput>` (same as A, marshalled to a child process) | One fully-reduced result per call, but the ONNX session lives in a worker process |

## Choosing an API

| If your model is… | And your host… | Use |
|---|---|---|
| A single forward pass (detection, classification, embedding) | Any | **A. One-shot** |
| Autoregressive (LLM/SLM token-by-token, stepwise diffusion) | Any | **B. Streaming** |
| Any model, but the host is a **GPU-rendered UI on AMD/ROCm** where in-process MIGraphX + Mesa/GL collide (`LLVM ERROR: inconsistency in registered CommandLine options` → `SIGABRT`) | Linux x64 + ROCm | **C. Out-of-process** |
| You want streaming *and* you hit the ROCm/GL collision | Linux x64 + ROCm UI | Compose: a **streaming** engine running in an **out-of-process** worker |

> The out-of-process motivation is **Linux/ROCm-specific** (the Mesa `libLLVM-19` ↔ comgr `libLLVM-20`
> collision documented in [`out-of-process-inference-engine-design.md`](out-of-process-inference-engine-design.md)),
> but the mechanism works on any platform — it is a generic "run the session in a child process"
> escape hatch.

---

## A. One-shot API

The original contract: `LoadAsync()` builds the ONNX session (with EP selection + warmup), then each
`PredictAsync(input)` runs one forward pass (`PreProcessWithContext` → `session.Run` →
`PostProcessWithContext`).

### Subclass pattern

```csharp
using InferenceEngine.Core;
using InferenceEngine.Core.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;

public sealed class MyDetector : BaseInferenceEngine<byte[], List<Detection>>
{
    public MyDetector(InferenceEngineOptions options, ILogger<MyDetector>? logger = null)
        : base(options, logger) { }

    protected override string ModelName => "my-model.onnx";

    protected override (List<NamedOnnxValue> InputTensors, object? Context)
        PreProcessWithContext(byte[] image)
    {
        // decode/resize/normalize → DenseTensor<float>; return (tensors, context-for-postproc)
        return (inputs, imageSize);
    }

    protected override List<Detection> PostProcessWithContext(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context)
    {
        // read output tensors, NMS, map boxes back to original image size
        return detections;
    }
}
```

### How load/predict/diagnostics work

- `LoadAsync()` is **idempotent** and auto-invoked by `PredictAsync` if needed. It resolves the model
  (registry auto-download via `ModelProviders`), selects the best execution provider, populates
  `InputMetadata`/`OutputMetadata`/`PrimaryInputName`, optionally warms up, then fires
  `OnSessionInitialized`.
- `EngineInfo` is an immutable snapshot: `Provider`, `IsFallback`, `FallbackReason`, and an ordered
  `Checks` trail (`/dev/kfd` → `libhsa-runtime64.so` → `libamdhip64.so.7` → `libmigraphx_c.so.3` →
  EP append → session construction on Linux; analogous on Windows/macOS).
- Wire diagnostics with a one-shot callback:

```csharp
var engine = new MyDetector(new InferenceEngineOptions
{
    UseGpuAcceleration = true,
    WarmupOnLoad = true,
    OnSessionInitialized = info => Console.WriteLine(
        $"Loaded {info.Provider} (fallback={info.IsFallback}, reason={info.FallbackReason})"),
});
await engine.LoadAsync();
var detections = await engine.PredictAsync(imageBytes);
```

### Reference example

[`../InferenceEngine.Examples/FaceDetectionEngine.cs`](../InferenceEngine.Examples/FaceDetectionEngine.cs)
— a complete `BaseInferenceEngine<byte[], List<PredictionResult>>` (YOLO11) with SkiaSharp
preprocessing and NMS postprocessing.

---

## B. Streaming API

For autoregressive generation: prefill over the prompt, then one decode pass per step with KV-cache
reuse. The engine yields **model-level steps** (a token id, a logit row, a text chunk) one at a time.

### Consuming a stream

```csharp
IStreamingInferenceEngine<string, string, string> engine = ...; // your engine

var cts = new CancellationTokenSource();
await foreach (var chunk in engine.PredictStreamingAsync(prompt, cancellationToken: cts.Token))
{
    Console.Write(chunk);
    if (userPressedStop) cts.Cancel();
}
```

- **Cancellation** is honored at each step boundary and aborts promptly, leaving the instance clean.
- **One-shot parity:** `await engine.PredictAsync(prompt)` runs the stream to completion and reduces
  it via `BuildFinalOutput` — one-shot consumers are unaffected.
- **Concurrency:** one in-flight stream per instance (KV-cache state is per-run); a second concurrent
  stream throws `InvalidOperationException`.

`StreamingOptions` is optional (defaults are safe/greedy): `MaxSteps`, `EosTokenIds`,
`Sampling` (`SamplingParams`: Temperature/TopK/TopP/RepetitionPenalty), `IncludePrefillInStream`,
`ChunkSize`.

```csharp
await foreach (var tok in engine.PredictStreamingAsync(
                   prompt,
                   new StreamingOptions { MaxSteps = 128, EosTokenIds = [2] },
                   ct))
{ ... }
```

### Implementing a streaming engine

Subclass `BaseStreamingInferenceEngine<TInput,TOutput,TStep>` and implement three hooks. The base
owns the decode loop (`BuildStepInputs → RunStep → ProcessStepOutputs → yield → dispose`), the stop
rules (`MaxSteps` + EOS), cancellation, the single-in-flight guard, per-step disposal, and greedy
sampling (`SampleNextToken`).

```csharp
public class MyGenerationEngine : BaseStreamingInferenceEngine<string, string, string>
{
    public MyGenerationEngine(InferenceEngineOptions options, ILogger<MyGenerationEngine>? logger = null)
        : base(options, logger) { }

    protected override string ModelName => "my-llm.onnx";

    // step == 0 → prefill (full prompt, empty past_key_values)
    // step  > 0 → decode (single token; feed prior present_key_values back as past_key_values)
    protected override (List<NamedOnnxValue> Inputs, object? BuildContext) BuildStepInputs(
        string prompt, object? state, int step)
    {
        // tokenize, assemble DenseTensor inputs, discover I/O names from InputMetadata.
        // Carry KV cache forward via CloneAsPastKeyValue(...).
        return (inputs, buildContext);
    }

    protected override (string Step, object? NextState, bool IsEos) ProcessStepOutputs(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? buildContext)
    {
        // read logits (+ present_key_values), sample (SampleNextToken(logits) for greedy),
        // detokenize one chunk, detect EOS against CurrentOptions.EosTokenIds,
        // return (chunk, nextKVState, isEos).
    }

    protected override string BuildFinalOutput(IReadOnlyList<string> steps) => string.Concat(steps);
}
```

To customize sampling (top-k/top-p/temperature/repetition-penalty), override
`protected virtual int SampleNextToken(ReadOnlySpan<float> logits)`. To carry the KV cache across
steps without it being disposed, use the `protected static NamedOnnxValue? CloneAsPastKeyValue(
DisposableNamedOnnxValue output, string pastName)` helper (covers `DenseTensor<float>`).

**Lifecycle is reused unchanged** via inheritance: `LoadAsync`, EP selection, warmup,
`ModelProviders`, and `EngineInfo` diagnostics all behave exactly as in the one-shot API.

### Reference example & spec

- [`../InferenceEngine.Examples/StreamingTextGenerationExample.cs`](../InferenceEngine.Examples/StreamingTextGenerationExample.cs)
  — a documented `BaseStreamingInferenceEngine<string,string,string>` skeleton (prefill/decode
  tensor assembly, KV-cache carry-back, sampling, EOS). Bring your own tokenizer.
- [`streaming-inference-guide.md`](streaming-inference-guide.md) — the full streaming consumer guide.

---

## C. Out-of-process API

Run the ONNX session in a **child worker process** over a length-prefixed JSON pipe (stdin/stdout).
The host-side proxy offers the **exact same contract** as the one-shot engine (`LoadAsync`/
`PredictAsync`/`EngineInfo`) but never loads `libonnxruntime`/`comgr`; the worker never loads Mesa/GL.
Two processes, two LLVMs, zero collision — and the consumer API is unchanged.

### Host-side wrapper

Subclass `OutOfProcessBaseInferenceEngine<TInput,TOutput>` (or wrap an instance). The base overrides
`LoadAsync`/`PredictAsync` to marshal to the worker and mirrors the worker's `EngineInfo` onto the
proxy via `ApplyLoadedState`, so diagnostics behave identically to the in-process engine.

```csharp
using System.Runtime.InteropServices;

public sealed class MyOutOfProcessDetector
    : OutOfProcessBaseInferenceEngine<byte[], List<Detection>>
{
    public MyOutOfProcessDetector(OutOfProcessEngineOptions options, ILogger? logger = null)
        : base(options, logger) { }

    // Optionally override serialization (default SystemTextJsonWorkerSerializer: byte[] → base64).
    // protected override IWorkerSerializer<byte[], List<Detection>> CreateSerializer() => ...;
}
```

`OutOfProcessEngineOptions` fields:

| Field | Purpose |
|---|---|
| `WorkerExecutablePath` (required) | Absolute path to the consumer's worker executable (your own app, launched with `WorkerArguments`). |
| `WorkerArguments` | Args to enter worker mode. Default `["--worker", "inference"]`; match your `Program.cs`. |
| `EngineId` | Consumer-defined id identifying which engine the worker should construct. |
| `WorkerEnvironment` | Env vars applied to the **worker only** (e.g. `HSA_OVERRIDE_GFX_VERSION=10.3.0`). Never applied to the host (which must stay GL-free of ROCm/comgr). |
| `EngineOptions` | In-process engine options mirrored to the worker (`ModelPath`, `UseGpuAcceleration`, `DeviceId`, `WarmupOnLoad`). |
| `StartupTimeout` | Time for the worker to start + handle `load` + emit `EngineInfo` (MIGraphX first compile can take ~85s; default 3 min). |
| `InferenceTimeout` | Per-`predict` round-trip timeout (default 60s). |

```csharp
var oop = new MyOutOfProcessDetector(new OutOfProcessEngineOptions
{
    WorkerExecutablePath = "/usr/local/bin/myapp",
    EngineId = "face-detection",
    WorkerEnvironment = new Dictionary<string, string>
    {
        ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0", // worker-only; host stays clean
    },
    EngineOptions = new InferenceEngineOptions
    {
        ModelPath = "/models/yolo11n.onnx",
        UseGpuAcceleration = true,
        WarmupOnLoad = true,
        OnSessionInitialized = info => Console.WriteLine(
            $"Worker loaded {info.Provider} (fallback={info.IsFallback})"),
    },
});
await oop.LoadAsync();                 // spawns worker, sends 'load', mirrors EngineInfo
var detections = await oop.PredictAsync(imageBytes);
```

### Worker entry point (`Program.cs`)

The consumer's worker executable calls `InferenceWorkerHost.RunAsync` with a factory that builds its
**own** engine (a normal `BaseInferenceEngine<TInput,TOutput>`, exactly like the one-shot path). The
loop reads `load`/`predict`/`shutdown` frames on stdin and writes `loaded`/`result` frames on stdout;
logging passes through on stderr.

```csharp
using InferenceEngine.Core;
using InferenceEngine.Core.Worker;

if (args.Contains("--worker"))
{
    await InferenceWorkerHost.RunAsync(static options =>
        new MyDetector(options));       // factory: build THIS process's engine
    return;
}
// ...otherwise: start the UI host as normal
```

The default serializer is `SystemTextJsonWorkerSerializer<TInput,TOutput>` (`byte[]` → base64, POCOs
→ JSON). Supply a custom `IWorkerSerializer<TInput,TOutput>` to `RunAsync` (worker) and override
`CreateSerializer()` (host) if your types need anything else.

### Opt-in wiring (Linux/ROCm only)

Only route through the worker when the collision is actually possible — a GL-rendering host on Linux
with the HIP runtime present. Everywhere else, the in-process engine is simpler and faster.

```csharp
static bool NeedsOutOfProcess() =>
    OperatingSystem.IsLinux()
    && NativeLibrary.TryLoad("libamdhip64.so.7", out _);

IInferenceEngine<byte[], List<Detection>> engine = NeedsOutOfProcess()
    ? new MyOutOfProcessDetector(oopOptions)
    : new MyDetector(inProcOptions);
```

### Design doc

[`out-of-process-inference-engine-design.md`](out-of-process-inference-engine-design.md) — the full
rationale, the Mesa `libLLVM-19` ↔ comgr `libLLVM-20` collision analysis, the IPC protocol
(`WorkerEnvelope` kinds: `load`/`loaded`/`predict`/`result`/`shutdown`), and the in-process remedies
that were tried and rejected.

---

## D. Ready-made text engines

For string-in/string-out work — embeddings and text generation — you do not need to write your own
engine subclass. The library ships two ready-made engines in `InferenceEngine.Core.Text` that own
tokenization and model-asset fetching end-to-end (via `Microsoft.ML.Tokenizers`):

| Engine | Built on | Yields |
|---|---|---|
| `TextEmbeddingEngine` | One-shot (`BaseInferenceEngine<string, float[]>`) | L2-normalized embeddings from all-MiniLM-L6-v2 |
| `TextGenerationEngine` | Streaming (`BaseStreamingInferenceEngine<string, string, string>`) | Streamed text chunks from Phi-3-mini-4k-instruct (cpu-int4) |

```csharp
using InferenceEngine.Core.Text;

var options = new InferenceEngineOptions { UseGpuAcceleration = false };

// Embeddings (one-shot)
await using var embed = new TextEmbeddingEngine(options);
await embed.LoadAsync();
float[] vector = await embed.PredictAsync("docker container");

// Generation (streaming — one-shot PredictAsync also works and returns the full text)
await using var gen = new TextGenerationEngine(options);
await foreach (string chunk in gen.PredictStreamingAsync("What is 2 + 2?"))
    Console.Write(chunk);
```

Both reuse the same lifecycle, provider selection, and diagnostics as the surfaces above. **Note:**
the Phi-3 int4 export is split across an `.onnx` graph and an `.onnx.data` sidecar — for real runs,
point `InferenceEngineOptions.ModelPath` at a complete local bundle with both files co-located.

### Detail

[`text-engines-guide.md`](text-engines-guide.md) — tokenizers, input/output tensor handling, the
full-recompute decode strategy, the `.onnx.data` limitation, and the tokenizer-asset cache.

---

## Diagnostics across all surfaces

`EngineInfo`, `IsFallback`, `FallbackReason`, the ordered `Checks` trail, and the
`OnSessionInitialized` callback **behave identically** across all surfaces. For the
out-of-process engine they are mirrored from the worker's `load` response, so a consumer cannot tell
from the diagnostics API whether the session is in-process or in a worker.

- `InferenceEngineInfo.Provider` — the EP that actually loaded (`MIGraphX`/`DirectML`/`CoreML`/`Cpu`).
- `InferenceEngineInfo.IsFallback` / `FallbackReason` — `None` on success, else a granular reason
  (`UserDisabledGpu`, `KfdDeviceMissing`, `HsaRuntimeUnloadable`, `HipRuntimeMissing`,
  `MIGraphXRuntimeMissing`, `ProviderAppendFailed`, `NativeLibraryLoadFailed`, …).
- `InferenceEngineInfo.Checks` — ordered `ProviderDiagnosticCheck` trail (Pass/Fail/Skip + detail).
- `InferenceEngineOptions.OnSessionInitialized` — fires once after the session is built (including
  fallback paths), on a thread-pool thread.

See [`linux-rocm-config.md`](linux-rocm-config.md) for the full ROCm checklist and
[`migraphx-rocblas-crash-troubleshooting.md`](migraphx-rocblas-crash-troubleshooting.md) for
crash triage.

---

## Example & spec locations

| Surface / feature | Reference example | Detail guide |
|---|---|---|
| A. One-shot | [`../InferenceEngine.Examples/FaceDetectionEngine.cs`](../InferenceEngine.Examples/FaceDetectionEngine.cs) | [`linux-rocm-config.md`](linux-rocm-config.md) |
| B. Streaming | [`../InferenceEngine.Examples/StreamingTextGenerationExample.cs`](../InferenceEngine.Examples/StreamingTextGenerationExample.cs) | [`streaming-inference-guide.md`](streaming-inference-guide.md) |
| C. Out-of-process | (your `Program.cs` `--worker` entry point + `OutOfProcessBaseInferenceEngine` subclass) | [`out-of-process-inference-engine-design.md`](out-of-process-inference-engine-design.md) |
| D. Text engines | `TextEmbeddingEngine` / `TextGenerationEngine` (`InferenceEngine.Core.Text`) | [`text-engines-guide.md`](text-engines-guide.md) |
| Package setup | — | [`../nuget.config`](../nuget.config) + [`../README.md`](../README.md) |

---

## Compatibility

All three surfaces are **additive**. Existing one-shot consumers — anything programmed to
`IInferenceEngine<TInput,TOutput>` or subclassing `BaseInferenceEngine<TInput,TOutput>` — compile and
behave unchanged. Streaming engines still satisfy the one-shot contract (one-shot `PredictAsync` is
routed through the stream and reduced via `BuildFinalOutput`); the out-of-process proxy implements
the same one-shot contract by marshalling. You can adopt a new surface without touching existing code.
