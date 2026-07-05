# Streaming Inference — Upstream Implementation Spec for `InferenceEngine.Core`

**Status:** Draft / upstream proposal
**Date:** 2026-07-04
**Intended upstream repository:** `intel-agency/inference-engine-lib` (see prerequisite)
**Target branch:** branch from `development` → open PR against `development`
**Proposed package impact:** additive only (new public types; no edits to existing public API)
**Derived from:** Appendix A of [`inference-engine-refactor-plan.md`](./inference-engine-refactor-plan.md),
expanded into a self-contained, implementation-ready spec.

---

## 0. Prerequisite — read first

> This spec was authored while the upstream source at
> `https://github.com/intel-agency/inference-engine-lib` was **not publicly reachable** (HTTP 404 at
> authoring time). All base-class API facts below were therefore taken from a *downstream*
> feasibility note that marks them as "verified," and the cited line numbers are from that note —
> **not** from a fresh read of the upstream `development` branch.
>
> **Before implementing anything, the upstream team must:**
> 1. Reconcile every type name, signature, accessibility modifier, and line number in this document
>    against the current `development` branch of `InferenceEngine.Core`.
> 2. Confirm the verified facts in §3 still hold (protected `_session`/`_runOptions`; public
>    `InputMetadata`/`OutputMetadata`/`PrimaryInputName`; virtual one-shot `PredictAsync`).
> 3. Treat any discrepancy as a blocker for §5/§6 and raise it before coding.

This document uses real `Microsoft.ML.OnnxRuntime` types for the loop body
(`InferenceSession.Run` → `IDisposableReadOnlyCollection<DisposableNamedOnnxValue>`;
`NamedOnnxValue.CreateFromTensor<T>(string, DenseTensor<T>)`), which the upstream library already
depends on.

---

## 1. Purpose, audience & non-goals

**Purpose.** Enable N-step autoregressive **streaming** generation (prefill + per-token decode with
KV-cache reuse) on top of the existing one-shot inference engine — exposed generically so any
consumer (LLM/SLM, speech, diffusion step-by-step, etc.) can stream model-level steps without
reimplementing the decode loop or the session/accel/warmup/model-download lifecycle.

**Audience.** The `InferenceEngine.Core` maintainers. This document is **self-contained**: a reviewer
who has never read any SupportAssistant plan must be able to implement streaming in
`InferenceEngine.Core` from this spec alone.

**Non-goals (deliberately out of the library):**
- **Tokenization / detokenization.** The library is and remains pure tensor-in / tensor-out. The
  streamed `TStep` is a model-level artifact (token id, logit row, or text chunk supplied by the
  engine). Text framing and decoding stay consumer concerns (see §12 and Appendix A.4).
- **Model download / caching.** Already solved by the existing `ModelProviders` machinery; reused
  unchanged.
- **A UI / rendering policy.** The consumer decides how to render or accumulate steps.

---

## 2. Background & motivation

The current public contract is strictly one-shot:

```
IInferenceEngine<TInput, TOutput>.PredictAsync(input)
    = PreProcessWithContext(input)        // build input tensors
    → _session.Run(inputs)               // EXACTLY ONE forward pass
    → PostProcessWithContext(outputs)    // reduce to TOutput
```

Autoregressive generation **cannot** be expressed through a single `session.Run`. It requires:

1. a **prefill** pass over the full prompt, and
2. a **decode** pass per generated token, where each step feeds the previous step's KV cache
   (`present_key_values`) back in as `past_key_values`, growing the attention mask / position ids,
   until an end-of-sequence token or a step budget is reached.

That is N forward passes with stateful feedback between them — fundamentally incompatible with the
single-`Run` pipeline. Today a consumer *can* author such a loop by hand because the session and
metadata are already accessible to subclasses (see §3); what is missing is a **generic, supported**
way to expose streaming so every consumer doesn't reinvent it.

---

## 3. Current contract recap (verified subclass access)

The following facts are what make a generic streaming base possible without touching the existing
types. (Line numbers per the downstream note; see §0 prerequisite.)

| Member (on `BaseInferenceEngine<TInput,TOutput>`) | Accessibility | Relevance |
| --- | --- | --- |
| `PredictAsync(TInput)` | `virtual`; calls `_session.Run` **once** | Override to route one-shot through the stream |
| `_session` | `protected readonly` | The subclass runs its own decode loop against it |
| `_runOptions` | `protected` | Per-`Run` options |
| `InputMetadata`, `OutputMetadata` | `public` | Discover exact I/O names/shapes at load |
| `PrimaryInputName` | `public` | Canonical first input |
| `ModelName`, `ModelDownloadUrls`, `ModelRegistry`, `ModelCacheDirectory` | overridable | Per-engine model registration, no upstream change |
| `LoadAsync`, provider selection, `WarmupOnLoad`, `ModelProviders`, `InferenceEngineInfo`, `OnSessionInitialized` | inherited | **Reused unchanged** by the new streaming base |

**Consequence:** a streaming base class can inherit `BaseInferenceEngine` and drive its own loop
against `_session`, reusing 100% of the lifecycle code, while leaving the one-shot contract intact.

---

## 4. Design goals & constraints

1. **Non-breaking / additive.** Zero edits to `IInferenceEngine<,>` or `BaseInferenceEngine<,>`.
   Only new types are introduced.
2. **One-shot behavior is preserved.** Existing consumers compile and behave identically.
3. **Consumer chooses.** On the same instance, a consumer may program to `IInferenceEngine`
   (one-shot), `IStreamingInferenceEngine` (streaming), or both.
4. **Reuse the lifecycle.** Provider selection, warmup, `ModelProviders`, `InferenceEngineInfo`, and
   `OnSessionInitialized` apply unchanged via inheritance.
5. **Model-agnostic loop skeleton; engine owns the contract.** Because HF-optimum ONNX exports vary
   wildly in I/O names, the **engine** assembles tensors (`BuildStepInputs`) and parses outputs
   (`ProcessStepOutputs`). The base supplies the loop skeleton, cancellation, sampling defaults, and
   stop rules.
6. **net10 native async.** `IAsyncEnumerable<T>` is native on net10.0 — no `Microsoft.Bcl.AsyncInterfaces`
   dependency is required for this repo's target.

---

## 5. Proposed public API (exact C# to implement)

### 5.1 New optional interface

A streaming engine **still is** a one-shot engine:

```csharp
namespace InferenceEngine.Core;

/// <summary>
/// An inference engine that can also yield intermediate model-level steps
/// (token ids / logits / chunks) as an asynchronous stream.
/// Implementations continue to satisfy the one-shot <see cref="IInferenceEngine{TInput,TOutput}"/>
/// contract; streaming is an additive capability.
/// </summary>
public interface IStreamingInferenceEngine<TInput, TOutput, TStep>
    : IInferenceEngine<TInput, TOutput>
{
    /// <summary>
    /// Yield one model-level step (token id / logit row / chunk) at a time.
    /// The consumer decides how to render or accumulate the steps.
    /// Cancellation is honored at each step boundary and inside the underlying session run.
    /// </summary>
    IAsyncEnumerable<TStep> PredictStreamingAsync(
        TInput input, CancellationToken cancellationToken = default);
}
```

### 5.2 New abstract base (reuses `BaseInferenceEngine` lifecycle)

```csharp
namespace InferenceEngine.Core;

/// <summary>
/// Reuses <see cref="BaseInferenceEngine{TInput,TOutput}"/> for session/accel/warmup/
/// model-download/diagnostics, and adds an autoregressive decode loop driven by
/// engine-supplied hooks. One-shot consumers see no change: <see cref="PredictAsync"/>
/// is routed through the streaming loop and reduced via <see cref="BuildFinalOutput"/>.
/// </summary>
public abstract class BaseStreamingInferenceEngine<TInput, TOutput, TStep>
    : BaseInferenceEngine<TInput, TOutput>, IStreamingInferenceEngine<TInput, TOutput, TStep>
{
    /// <summary>One-shot behavior is preserved: run the stream to completion and reduce it.</summary>
    public override async Task<TOutput> PredictAsync(TInput input)
    {
        var steps = new List<TStep>();
        await foreach (var s in PredictStreamingAsync(input))
            steps.Add(s);
        return BuildFinalOutput(steps);
    }

    /// <summary>Drives prefill (step 0) → decode (step &gt; 0) → stop.</summary>
    public abstract IAsyncEnumerable<TStep> PredictStreamingAsync(
        TInput input, CancellationToken cancellationToken = default);

    // ---- Model-aware hooks: the engine knows its own I/O contract. -----------------

    /// <summary>
    /// Build the ONNX inputs for a given step and return the carry-forward state.
    /// <paramref name="step"/> == 0 means prefill (full prompt); step &gt; 0 means decode
    /// (feed previous <c>present_key_values</c> back as <c>past_key_values</c>).
    /// </summary>
    protected abstract (List<NamedOnnxValue> Inputs, object? State) BuildStepInputs(
        TInput input, object? state, int step);

    /// <summary>
    /// Parse one forward pass's outputs into a streamed step plus the next carry-forward state,
    /// and signal end-of-sequence.
    /// </summary>
    protected abstract (TStep Step, object? NextState, bool IsEos) ProcessStepOutputs(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? state);

    /// <summary>Sampling strategy. Greedy by default; override for top-k/top-p/temperature/
    /// repetition-penalty using <see cref="StreamingOptions.Sampling"/>.</summary>
    protected virtual TStep SampleNextToken(/* logits + StreamingOptions */) =>
        throw new NotImplementedException("Greedy default sampling is implemented by the base; " +
            "see implementation note in §6. Override to customize.");

    /// <summary>Reduce the accumulated steps into the final one-shot <typeparamref name="TOutput"/>.</summary>
    protected abstract TOutput BuildFinalOutput(IReadOnlyList<TStep> steps);
}
```

**Why this shape works:** because `BaseStreamingInferenceEngine` **inherits**
`BaseInferenceEngine`, every piece of lifecycle code (`LoadAsync`, provider selection, warmup,
`ModelProviders`, `InferenceEngineInfo`, `OnSessionInitialized`) is reused unchanged. `IInferenceEngine`
one-shot behavior is preserved by routing `PredictAsync` through the streaming loop. A consumer can
program to **either** `IInferenceEngine` (one-shot) **or** `IStreamingInferenceEngine` (streaming)
on the same instance.

> **Implementation note (for the maintainer).** The base should own the loop body
> (`for step in 0..MaxSteps { BuildStepInputs → _session.Run → ProcessStepOutputs → yield → dispose }`)
> so subclasses only supply the four hooks. The signature of `SampleNextToken` must be finalized
> against the real logits type used by the base; the placeholder above is illustrative — see §6.

---

## 6. Behavioral contract (the "exactly how")

This section is normative. The upstream implementation and its tests must satisfy every bullet.

### 6.1 Prefill (step 0)
- Prompt tensor assembly is performed by the engine in `BuildStepInputs(input, state: null, step: 0)`.
- The first forward pass is a **full-sequence** pass over the entire prompt.

### 6.2 Decode (step > 0)
- Feed the previous step's `present_key_values` outputs back as `past_key_values` inputs.
- Grow the `attention_mask` by one position per generated token.
- Also grow `position_ids` / `cache_position` if the export requires them (the engine decides in
  `BuildStepInputs` based on `InputMetadata`).
- Exactly **one** token is produced and yielded per decode step.

### 6.3 KV-cache plumbing helper (protected)
- The base supplies the **model-agnostic** part of carrying `present_*` outputs back as `past_*`
  inputs and growing the mask/position tensors.
- Because HF-optimum ONNX exports vary in I/O names, the **engine** owns concrete tensor assembly
  (`BuildStepInputs`) and output parsing (`ProcessStepOutputs`). The base never hardcodes an export's
  names.

### 6.4 Stopping rules
- Stop when the sampled token is in `StreamingOptions.EosTokenIds`, **or** when `MaxSteps` is reached.
- Rules are evaluated in the loop after `ProcessStepOutputs` returns `IsEos`, and also against the
  step count.

### 6.5 Sampling
- `SampleNextToken` defaults to **greedy** (argmax).
- It is `virtual` so an engine can implement top-k / top-p / temperature / repetition-penalty using
  `StreamingOptions.Sampling`.
- Finalize the logits-typed signature against the real base; greedy must be deterministic.

### 6.6 Cancellation
- Thread the `CancellationToken` into `_session.Run` (where supported) **and** into the
  `IAsyncEnumerable` step boundary.
- On cancellation: raise `OperationCanceledException`; **discard partial output**; leave no
  half-populated KV-cache state that could leak into a subsequent run on the same instance.

### 6.7 Parity invariant (load-bearing)
- For identical input and options:
  `BuildFinalOutput(PredictStreamingAsync(input))` **must equal** `PredictAsync(input)`.
- This is the single most important correctness guarantee and must be covered by a test (§10).

### 6.8 Resource disposal
- `Dispose()` each step's `IDisposableReadOnlyCollection<DisposableNamedOnnxValue>` **after**
  `ProcessStepOutputs` returns, on every path (including exception and cancellation).

### 6.9 Concurrency / reentrancy
- One engine instance supports **one in-flight stream at a time**. KV-cache state is per-run; a
  second concurrent stream on the same instance must be rejected (recommend
  `InvalidOperationException`).
- A one-shot `PredictAsync` and a `PredictStreamingAsync` must not interleave on the same instance.

---

## 7. Configuration surface (exact C# to implement)

```csharp
namespace InferenceEngine.Core;

/// <summary>Per-call streaming controls. Defaults are safe/greedy.</summary>
public sealed class StreamingOptions
{
    /// <summary>Maximum decode steps before the loop stops.</summary>
    public int MaxSteps { get; set; } = 256;

    /// <summary>Token ids that terminate generation.</summary>
    public IReadOnlyList<int> EosTokenIds { get; set; } = Array.Empty<int>();

    /// <summary>Sampling strategy (greedy by default).</summary>
    public SamplingParams Sampling { get; set; } = new();

    /// <summary>If true, also yield prompt tokens from the prefill pass. Default false.</summary>
    public bool IncludePrefillInStream { get; set; } = false;

    /// <summary>Batch >=1 decoded steps per yield (null/1 = one step per yield). Default null.</summary>
    public int? ChunkSize { get; set; } = null;
}

/// <summary>Sampling parameters. Disabled/identity values mean "off".</summary>
public sealed class SamplingParams
{
    /// <summary>0 => greedy (argmax).</summary>
    public float Temperature { get; set; } = 0f;

    /// <summary>0 => disabled.</summary>
    public int TopK { get; set; } = 0;

    /// <summary>1 => disabled.</summary>
    public float TopP { get; set; } = 1f;

    /// <summary>1 => disabled.</summary>
    public float RepetitionPenalty { get; set; } = 1f;
}
```

> **Design note.** `StreamingOptions` should be passable to `PredictStreamingAsync` via an
> **overload** (recommended) with a default of `null` → use instance-level options. Pick one shape
> and apply it consistently across `PredictStreamingAsync`, `PredictAsync`, and `SampleNextToken`.

---

## 8. Non-breaking / compatibility analysis

- **No changes** to `IInferenceEngine<TInput,TOutput>` or `BaseInferenceEngine<TInput,TOutput>`.
- Only **new** public types are added: `IStreamingInferenceEngine<,,>`,
  `BaseStreamingInferenceEngine<,,>`, `StreamingOptions`, `SamplingParams`.
- Existing one-shot consumers compile and behave identically.
- `BaseStreamingInferenceEngine` derives from `BaseInferenceEngine`, so provider selection, warmup,
  `ModelProviders`, `InferenceEngineInfo`, and `OnSessionInitialized` all apply unchanged.
- SemVer: this is an **additive minor** (new capabilities), not a breaking change.

---

## 9. Implementation task list (ordered, for the upstream team)

1. Add `IStreamingInferenceEngine<TInput,TOutput,TStep>` (new file).
2. Add `BaseStreamingInferenceEngine<TInput,TOutput,TStep>` implementing the loop skeleton:
   - `PredictAsync` routed through the stream,
   - `PredictStreamingAsync` driving prefill → decode → stop,
   - per-step `Dispose` of outputs,
   - cancellation threading.
3. Add the protected KV-cache plumbing helper (`present_*` → `past_*`; grow mask/positions) —
   model-agnostic; engine supplies names.
4. Add `StreamingOptions` + `SamplingParams`; wire `SampleNextToken` greedy default + overrides.
5. Thread `CancellationToken` into `_session.Run` and the `IAsyncEnumerable`.
6. Implement stop rules (EOS + `MaxSteps`).
7. Implement the single-in-flight-stream guard (§6.9).
8. Write the tests in §10.
9. Write the docs/example in §11.
10. Build under the repo's `TreatWarningsAsErrors=true` gate; keep the existing suite green.

---

## 10. Test requirements (must-pass acceptance)

- **Parity:** `BuildFinalOutput(PredictStreamingAsync(input)) == PredictAsync(input)` for identical
  input + options. *(§6.7 — most important.)*
- **Cancellation:** a mid-stream `CancellationToken` aborts promptly; the instance is left clean
  (a subsequent run behaves correctly; no leaked KV-cache state).
- **KV-cache correctness:** streaming-with-KV output equals full-recompute-per-step output (validates
  that carrying `present_*` back as `past_*` is correct).
- **Stop rules:** an EOS token stops the loop; `MaxSteps` stops the loop.
- **Concurrency:** a second call on a busy instance is rejected per §6.9.
- **Determinism:** greedy sampling is deterministic across repeated runs.
- **Disposal:** outputs are disposed on the success, exception, and cancellation paths (no leak).
- **Regression:** existing one-shot tests still pass; build is clean under
  `TreatWarningsAsErrors=true`.

---

## 11. Documentation requirements

- **README:** add a "Streaming inference" section showing the `IStreamingInferenceEngine` interface
  and a minimal subclass sketch.
- **Example:** add `StreamingTextGenerationExample` mirroring the existing `FaceDetectionEngine`
  example — subclass `BaseStreamingInferenceEngine<,,>`, implement the three hooks
  (`BuildStepInputs`, `ProcessStepOutputs`, `BuildFinalOutput`), and consume the stream with
  `await foreach`.

---

## 12. Consumer perspective & migration (optional read for maintainers)

This section is **not** part of the upstream API. It exists to motivate the proposal and to give the
upstream team a concrete validation consumer. It is clearly separated; the core team may skip it.

**The consumer.** SupportAssistant is building an on-device Phi-3 SLM engine. Today (per the local
plan) it must hand-write the decode loop:

```csharp
// BEFORE (local, hand-written loop) — from the SupportAssistant refactor plan, Phase 3
public class Phi3GenerationEngine : BaseInferenceEngine<string, string>
{
    public override Task<string> PredictAsync(string prompt)
    {
        // tokenize → prefill (_session.Run) → decode loop with KV-cache carry-back
        //   → sample → grow mask → _session.Run → stop on EOS/MaxTokens → detokenize
    }
    public IAsyncEnumerable<string> StreamAsync(string prompt, CancellationToken ct) { /* same loop, yield */ }
}
```

**After this proposal lands upstream**, the same engine collapses to a
`BaseStreamingInferenceEngine<string,string,string>` subclass with three small hooks — a clean
migration with **no one-shot regression** (the parity test in §10 gates it):

```csharp
// AFTER (upstream streaming base merged)
public class Phi3GenerationEngine : BaseStreamingInferenceEngine<string, string, string>
{
    protected override (List<NamedOnnxValue> Inputs, object? State) BuildStepInputs(
        string prompt, object? state, int step)
    {
        // step == 0: tokenize prompt, build prefill tensors (with empty past_key_values)
        // step  > 0: build single-token decode tensors, feed prior present_key_values
    }

    protected override (string Step, object? NextState, bool IsEos) ProcessStepOutputs(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? state)
    {
        // read logits (+ present_key_values) → SampleNextToken → detokenize one chunk
        // carry present_key_values forward; detect EOS
    }

    protected override string BuildFinalOutput(IReadOnlyList<string> steps) =>
        string.Concat(steps);

    // Optional: override SampleNextToken for top-p/temperature.
}
```

This is the value of the generic base: every consumer stops reinventing the decode loop, and the
library keeps its pure tensor-in/out character because tokenization/detokenization stay in the
engine hooks (per Appendix A.4).

Cross-reference: local implementation context lives in
[`inference-engine-refactor-plan.md`](./inference-engine-refactor-plan.md), Phase 3 and Appendix A.

---

## 13. Open questions / risks (for the upstream team)

- **KV-cache I/O variance across HF-optimum ONNX exports:** exact `past_key_values.*` ↔
  `present_key_values.*` names, `use_cache_branch` toggles, and whether prefill + decode share one
  graph or require two. *Mitigation:* the engine owns these via `BuildStepInputs` /
  `ProcessStepOutputs`; the base stays export-agnostic.
- **Full-recompute fallback:** if an export lacks a KV-cache graph, the consumer can fall back to
  full-recompute-per-step (O(n²), acceptable for short contexts). Worth documenting as a known
  pattern, not implementing in the base.
- **Signature reconciliation:** due to the unreachable upstream source (§0), the maintainer must
  confirm every signature against `development` before coding.
- **Concurrency model:** confirm the single-in-flight-stream rule (§6.9) is acceptable, or define
  per-call isolated state (larger change — out of scope here).
- **`SampleNextToken` logits type:** must be pinned to the real type the base exposes.

---

## 14. Acceptance checklist (sign-off for this spec)

- [ ] Self-contained — passes the audience test in §1.
- [ ] Exact C# for the interface (§5.1), the base (§5.2), and the options (§7) included.
- [ ] Behavioral contract (§6) covers prefill, decode, KV-cache carry-back, stop rules, sampling,
      cancellation, parity, disposal, and concurrency.
- [ ] §0 Prerequisite (reconcile vs `development`) and the upstream-404 caveat are present.
- [ ] Test requirements (§10) and acceptance criteria are enumerated.
- [ ] Non-breaking proof (§8) is stated.
- [ ] Cross-links to Appendix A and the refactor plan are present.

---

*Relationship to Appendix A:* Appendix A of
[`inference-engine-refactor-plan.md`](./inference-engine-refactor-plan.md) is the short feasibility
summary that motivated this document. This spec is its implementation-ready expansion for the
upstream team.
