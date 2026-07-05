# Streaming Inference — Consumer Implementation Guide

> **Audience:** developers building a consuming application (an LLM/SLM runner, a speech engine, a
> step-by-step diffusion pipeline) on top of `InferenceEngine.Core`. This guide is self-contained.

## What it is

Streaming inference lets an engine yield intermediate **model-level steps** (a token id, a logit row,
or a text chunk) one at a time as an `IAsyncEnumerable<TStep>`, instead of producing a single
fully-reduced result from one forward pass. It is what you need for **autoregressive generation**
(prefill over the prompt, then one decode pass per generated token with KV-cache reuse).

Crucially, streaming is **additive**: a streaming engine *still is* a one-shot engine. On the same
instance a consumer may program to `IInferenceEngine<TInput,TOutput>` (one-shot),
`IStreamingInferenceEngine<TInput,TOutput,TStep>` (streaming), or both.

## Package & version

The streaming types ship in **`InferenceEngine.Core` 1.1+**. Reference the package and you get:

| Type | Purpose |
|---|---|
| `IStreamingInferenceEngine<TInput,TOutput,TStep>` | The streaming contract (adds `PredictStreamingAsync`). |
| `BaseStreamingInferenceEngine<TInput,TOutput,TStep>` | Abstract base that owns the decode loop; you implement three hooks. |
| `StreamingOptions`, `SamplingParams` | Per-call controls (max steps, EOS ids, sampling). |

```xml
<PackageReference Include="InferenceEngine.Core" Version="1.1.0-dev.13" />
```

## Consuming a stream

```csharp
IStreamingInferenceEngine<string, string, string> engine = ...; // your engine

var cts = new CancellationTokenSource();
await foreach (var chunk in engine.PredictStreamingAsync(prompt, cancellationToken: cts.Token))
{
    Console.Write(chunk);          // render/accumulate as you like
    if (userPressedStop) cts.Cancel();
}
```

- **Cancellation** is honored at each step boundary and aborts promptly, leaving the instance clean
  (a subsequent call works — no leaked KV-cache state).
- **One-shot parity:** `await engine.PredictAsync(prompt)` returns exactly
  `BuildFinalOutput(collected stream)` — so one-shot consumers are unaffected.
- **Concurrency:** one in-flight stream per engine instance. A second concurrent stream throws
  `InvalidOperationException` (KV-cache state is per-run).

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

## Implementing a streaming engine

Subclass `BaseStreamingInferenceEngine<TInput,TOutput,TStep>` and implement three hooks. The base
owns the loop (`BuildStepInputs → RunStep → ProcessStepOutputs → yield → dispose`), the stop rules
(`MaxSteps` + EOS), cancellation, the single-in-flight guard, per-step disposal, and greedy sampling.

```csharp
public class MyGenerationEngine : BaseStreamingInferenceEngine<string, string, string>
{
    public MyGenerationEngine(InferenceEngineOptions options, ILogger<MyGenerationEngine>? logger = null)
        : base(options, logger) { }

    protected override string ModelName => "my-model.onnx";

    // step == 0 → prefill (full prompt, empty past_key_values)
    // step  > 0 → decode (single token; feed prior present_key_values back as past_key_values)
    protected override (List<NamedOnnxValue> Inputs, object? BuildContext) BuildStepInputs(
        string prompt, object? state, int step)
    {
        // tokenize, assemble DenseTensor inputs, discover I/O names from InputMetadata.
        // Carry KV cache forward via BaseStreamingInferenceEngine.CloneAsPastKeyValue(...).
        return (inputs, buildContext);
    }

    protected override (string Step, object? NextState, bool IsEos) ProcessStepOutputs(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? buildContext)
    {
        // read logits (+ present_key_values), sample (use SampleNextToken(logits) for greedy),
        // detokenize one chunk, detect EOS against CurrentOptions.EosTokenIds,
        // return (chunk, nextKVState, isEos).
    }

    protected override string BuildFinalOutput(IReadOnlyList<string> steps) => string.Concat(steps);
}
```

**Everything lifecycle is reused unchanged** via inheritance: `LoadAsync`, hardware provider
selection (MIGraphX on Linux/DirectML on Windows/CoreML on macOS with graceful CPU fallback), warmup,
`ModelProviders` (registry auto-download), and the diagnostics API. To customize sampling
(top-k/top-p/temperature), override `protected virtual int SampleNextToken(ReadOnlySpan<float> logits)`.

## Diagnostics integration (recommended)

Wire `InferenceEngineOptions.OnSessionInitialized` to learn which provider actually loaded (e.g.
`MIGraphX` vs `Cpu` + a granular `FallbackReason`), and read `engine.EngineInfo.Checks` for the
ordered provider-selection trail. This is exactly how InferenceApp surfaces "Detect: Cpu (fallback:
HipRuntimeMissing)" in its footer and `--diag inference` CLI. See
[`linux-rocm-config.md`](linux-rocm-config.md) and the InferenceApp Section-B implementation.

## Where to find the example

- **Reference skeleton:** [`InferenceEngine.Examples/StreamingTextGenerationExample.cs`](../InferenceEngine.Examples/StreamingTextGenerationExample.cs)
  — a documented `BaseStreamingInferenceEngine<string,string,string>` showing prefill/decode tensor
  assembly, KV-cache carry-back (`CloneAsPastKeyValue`), sampling, and EOS detection. It is a
  **skeleton**: bring a real HF-optimum ONNX export + a tokenizer (tokenization is a deliberate
  non-goal of the library — the engine stays pure tensor-in/tensor-out).
- **Full design spec:** [`plan_docs/inference-engine-streaming-upstream-spec.md`](../plan_docs/inference-engine-streaming-upstream-spec.md)
  — the normative behavioral contract (prefill, decode, KV-cache, stop rules, cancellation, parity,
  disposal, concurrency), test requirements, and open questions.
- **README:** a short "Streaming inference" section in [`README.md`](../README.md).

## Non-goals (by design)

- **Tokenization / detokenization** — stays in your engine/consumer; the library is tensor-in/out.
- **Model download/caching** — already handled by `ModelProviders`; reused unchanged.
- **A UI/rendering policy** — your consumer decides how to render or accumulate steps.
- **Non-greedy sampling in the base** — greedy argmax is provided; override `SampleNextToken` for the rest.

## Limitations

- One in-flight stream per instance (reentrant streams throw `InvalidOperationException`).
- The KV-cache carry helper (`CloneAsPastKeyValue`) covers `DenseTensor<float>`; other dtypes need
  extension. HF-optimum export I/O names vary — your engine owns concrete tensor assembly.
