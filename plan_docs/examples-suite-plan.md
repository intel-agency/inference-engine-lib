# Example-App Suite for InferenceEngine.Core — Implementation Plan

- **Date:** 2026-10-24
- **Status:** Plan written; implementation **not yet approved** (user authorized writing this document only)
- **Scope repo:** `intel-agency/inference-engine-lib` (branch `feature/examples-suite` off `development` when implementation is approved)

## Goal

A suite of simple, runnable example apps — one per base inference engine type that consumers
can extend in `InferenceEngine.Core` — each paired with a domain that engine type is commonly
used for (modeled on the existing `FaceDetectionEngine` example), with tests for every example
and collective coverage of the library's full feature surface.

## Where this lives and why

All work happens in **`intel-agency/inference-engine-lib`** — that is where
`InferenceEngine.Core`, the `InferenceEngine.Examples` project, and the "FaceIdExample"
(`FaceDetectionEngine`) live. The lib repo's own `AGENTS.md` governs (XML docs on public
members, xUnit + `SkippableFact`, coverage baseline must not drop, `./scripts/agent-self-check.sh`
gate, signed commits, PRs into `development`). Current checkout is on `ci/install-cosign`;
implementation starts by branching `feature/examples-suite` off `development`.

## What exists today (verified in code)

Three base engine types in `InferenceEngine.Core` — the "types you can extend":

1. **`BaseInferenceEngine<TInput,TOutput>`** — one-shot ONNX engine (session lifecycle, provider
   selection MIGraphX/DirectML/CoreML→CPU fallback, warmup, model download, diagnostics).
   Extended by `FaceDetectionEngine` (YOLO, SkiaSharp) and `TextEmbeddingEngine` (MiniLM) today.
2. **`BaseStreamingInferenceEngine<TInput,TOutput,TStep>`** — autoregressive
   `IAsyncEnumerable` decode loop (3 hooks, KV-cache helpers, greedy/temperature/top-k/top-p
   sampler). Extended by `TextGenerationEngine` (Phi-3) and a throwing
   `StreamingTextGenerationExample` skeleton.
3. **`OutOfProcessBaseInferenceEngine<TInput,TOutput>`** — host-side proxy marshalling to a
   child worker process (`InferenceWorkerHost` loop, length-prefixed JSON pipe, status
   mirroring, worker env overrides, timeouts). **Has no example at all today.**

Supporting feature surface to cover: `InferenceEngineOptions` (GPU toggle, DeviceId,
WarmupOnLoad, ModelPath/ModelProvider, MIGraphXModelCacheDirectory, MaxModelDownloadBytes,
OnStatusChanged, OnSessionInitialized), ModelProviders (Local/Downloading/Registry +
`models.json` + sha256), Diagnostics (`InferenceEngineInfo`, ordered `Checks`, fallback
reasons), `StreamingOptions` (MaxSteps, EosTokenIds, Sampling, IncludePrefillInStream,
ChunkSize), built-in text engines + `TokenizerAssets`, `IWorkerSerializer` seam.

## Design

**Structure** — engines stay testable library code; apps are thin entry points:

- `InferenceEngine.Examples/` (existing lib) gains new example engines + a shared
  `ConsoleDiagnostics` helper (status/EngineInfo/check printing all apps reuse — itself
  example code for consuming diagnostics).
- New **`examples/`** solution folder with four console apps (`OutputType=Exe`, net10.0,
  `IsPackable=false`, host-RID `RuntimeIdentifier` pattern copied from existing csproj files,
  **plus the Linux native-lib symlink / Windows-lib cleanup MSBuild targets duplicated from
  `InferenceEngine.Tests.csproj`** — those targets operate per-project output and do not flow
  transitively):

| App | Base type demonstrated | Domain | Engine |
|---|---|---|---|
| `DetectionExample` | `BaseInferenceEngine` | object detection (vision) | existing `FaceDetectionEngine` (yolo11n) |
| `EmbeddingsExample` | one-shot consumption of a built-in engine | semantic search (rank corpus vs query by cosine similarity) | built-in `TextEmbeddingEngine` (MiniLM) |
| `ChatExample` | `BaseStreamingInferenceEngine` | LLM chat / text generation | **new** `Gpt2StreamingGenerationEngine` in Examples lib |
| `WorkerExample` | `OutOfProcessBaseInferenceEngine` + `InferenceWorkerHost` | crash-isolated GPU detection worker | **new** `WorkerDetectionProxy` in Examples lib hosting `FaceDetectionEngine` in a child process |

CLI flags across apps collectively exercise every options knob:
`--cpu/--device/--model-path/--no-warmup/--max-download-mb/--mxr-cache/--status/--temperature/--top-k/--top-p/--max-steps/--one-shot/...`.

**Key design decisions** (with rationale):

- **Chat model = single-file GPT-2-family decoder (distilgpt2-class, ~260 MB) fetched via
  `ModelDownloadUrls` + models.json entry with pinned sha256**, full-recompute decode (same
  strategy as Core's `TextGenerationEngine` — model-agnostic, no KV-cache naming fragility),
  GPT-2 byte-level BPE via `Microsoft.ML.Tokenizers` 2.0 (`BpeTokenizer`), EOS 50256. Chosen
  over the built-in Phi-3 because the Phi-3 cpu-int4 export needs a manually-assembled
  `.onnx.data` bundle (bad out-of-box DX). ChatExample also gets `--phi3 <dir>` to run the
  built-in `TextGenerationEngine` from a local bundle, so the built-in streaming engine is
  demonstrated live too. Exact tokenizer factory + model URL verified against
  Microsoft.ML.Tokenizers 2.0 / HF during implementation; the skeleton
  `StreamingTextGenerationExample` stays (docs reference it).
- **WorkerExample re-enters itself** (`--worker inference`, mirroring
  `OutOfProcessEngineOptions` defaults and the InferenceApp production pattern): default mode
  spawns its own executable as worker, mirrors worker `EngineInfo`/status, demonstrates
  `WorkerEnvironment` passthrough, timeouts, graceful shutdown, and a custom
  `IWorkerSerializer` for the `PredictionResult` DTOs.
- **`SamplingParams.RepetitionPenalty` is declared but not implemented by the base sampler —
  it is NOT demonstrated** (no inert flags); noted as a follow-up issue.

## Milestones (each independently shippable; gate = `./scripts/agent-self-check.sh` + coverage baseline maintained)

**M0 — Scaffolding.** Branch off `development`; create the four app projects + slnx
`/examples/` folder + per-RID lockfiles committed under `lockfiles/` (linux-x64 leg, matching
existing pattern); shared `ConsoleDiagnostics` helper; placeholder Programs. Self-check green.

**M1 — DetectionExample.** Full CLI over the existing engine; prints detections +
provider/diagnostics trail. Tests: arg-parse→`InferenceEngineOptions` mapping, output
formatting (hermetic); network-gated E2E already exists for the engine — add app-logic level
E2E gated on network + native-lib loadable (`bus.jpg` person/bus/car).

**M2 — EmbeddingsExample.** Cosine-similarity helper + corpus ranking over built-in
`TextEmbeddingEngine`. Tests: cosine math hermetic; network-gated ordering test
(`sim(query, relevant) > sim(query, unrelated)`).

**M3 — ChatExample + `Gpt2StreamingGenerationEngine`.** Engine (tokenizer fetch mirroring
`TokenizerAssets` pattern via public `DownloadingModelProvider`, tokenizer test seam like
Core's `TestTokenizer`); app = interactive REPL + `--one-shot` (demonstrates
`BuildFinalOutput` path) + streaming flags + Ctrl+C cancellation; models.json entry. Tests:
hermetic (tokenizer seam, prefill/decode tensor assembly shapes+dtypes, logits→token→EOS
decode via fake outputs, final-output reduction, sampling on crafted logits); gated real
generation (network+ORT, `MaxSteps=8`, non-empty text).

**M4 — WorkerExample + `WorkerDetectionProxy`.** Self-spawn host/worker app as described.
Tests: serializer round-trip hermetic; injected-streams protocol tests
(`SetWorkerStreamsForTesting` precedent); real self-spawn E2E gated on built binaries +
network (pattern from `OutOfProcessIpcFaultTests` real-exe test).

**M5 — Docs, coverage, CI.** README "Examples" section (app ↔ base type ↔ domain ↔ features
matrix), `examples/README.md`, `docs/examples-guide.md` + toc entry, CHANGELOG. Verify
coverage report shows no baseline regression (Examples code is counted repo-wide), push
branch, open PR to `development` (signed commits), monitor `build-test.yml` (3-OS matrix) +
`rocm-gpu-test.yml` (self-hosted GPU leg — MIGraphX-gated example tests skip cleanly on
hosted runners) to green.

## Feature-coverage matrix (suite must demonstrate every library feature)

| Library feature | Demonstrated in |
|---|---|
| `IInferenceEngine` contract (`LoadAsync`/`PredictAsync`) | all apps |
| `BaseInferenceEngine` extension (Pre/PostWithContext, context passing) | Detection (existing engine), Chat engine (inherits) |
| `ModelName` / `ModelDownloadUrls` / `ModelRegistry` overrides | Chat engine (URLs override), Detection (registry lookup), new models.json entry |
| `LocalModelProvider` via `ModelPath` | Detection `--model-path` |
| `DownloadingModelProvider` (cache, fallback URLs, sha256, `MaxModelDownloadBytes`) | all downloading apps; `--max-download-mb` flag |
| `UseGpuAcceleration` / `DeviceId` | Detection/Worker flags |
| `WarmupOnLoad` | `--no-warmup` flag |
| `MIGraphXModelCacheDirectory` | `--mxr-cache` flag (linux) |
| `OnStatusChanged` + `InferenceEngineStatus`/`Detail` | shared console status printer in all apps |
| `OnSessionInitialized` + `EngineInfo`/`Checks`/fallback reason | shared diagnostics printer |
| Input/Output metadata inspection (dynamic shapes, `PrimaryInputName`) | Detection (dynamic shape fallback), Chat |
| `BaseStreamingInferenceEngine` extension (3 hooks) | Chat engine |
| `PredictStreamingAsync` + `StreamingOptions` (MaxSteps, EosTokenIds) | Chat app |
| Sampling (Temperature/TopK/TopP) | Chat app flags |
| `IncludePrefillInStream` / `ChunkSize` | Chat app flags |
| Cancellation at step boundaries | Chat Ctrl+C |
| One-shot `PredictAsync` on a streaming engine (`BuildFinalOutput`) | Chat `--one-shot` |
| `TextEmbeddingEngine` + `TokenizerAssets` | Embeddings app |
| `TextGenerationEngine` (built-in) | Chat `--phi3 <dir>` from a local bundle |
| `OutOfProcessBaseInferenceEngine` + `InferenceWorkerHost` + protocol | Worker app |
| `WorkerEnvironment` / timeouts / status mirroring | Worker app flags |
| `IWorkerSerializer` customization seam | Worker app (custom serializer for DTOs) |
| CPU fallback + ordered diagnostic trail | every app's provider printout |
| Disposal (`IDisposable`/`IAsyncDisposable`) | all apps |
| `SamplingParams.RepetitionPenalty` | **not demonstrated** — declared but unimplemented in the base sampler; follow-up issue |

## Validation & risks

- Gate per milestone: `./scripts/agent-self-check.sh` (lint/build/test) in the lib repo;
  `dotnet test --collect:"XPlat Code Coverage"` to confirm baseline.
- Hermetic tests need no network/GPU (skippable-fact capability probes follow
  `FaceDetectionEngineTests` exactly), so the 3-OS CI matrix stays green; real-model tests
  skip when offline.
- Risks: (1) distilgpt2 ONNX URL/tokenizer API details — resolved in M3 with a
  pinned-sha256 fallback URL pair; (2) symlink MSBuild targets per app — copied from Tests
  verbatim, low risk; (3) large model downloads in tests — all gated, never in hermetic
  paths.
