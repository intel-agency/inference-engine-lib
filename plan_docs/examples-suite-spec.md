# Example-App Suite — Implementation Specification

- Date: 2026-10-24
- Status: Draft specification; implementation not approved
- Source: [examples-suite-plan.md](examples-suite-plan.md)
- Scope: intel-agency/inference-engine-lib only

## 1. Authorization, goals, and non-goals

**REQ-SCOPE-01 — Authorization.** This document specifies future work; it does not approve implementation.
The present deliverable is this specification plus one README discoverability link, not executable examples.
Preserve the source plan unchanged. Its `ci/install-cosign` checkout statement is historical context, not a new repository-state assertion.
All design choices marked **Proposed** below remain subject to implementation approval and the verification gates in section 13.

**REQ-SCOPE-02 — Goal.** Provide four small runnable examples covering the three extensible engine bases, built-in text-engine consumption, and their collective feature surface.
Use familiar object detection, semantic search, text generation, and crash-isolated detection domains; test every app and new engine/helper.
Preserve the original plan's M0–M5 scope, feature matrix, resource lifecycle, documentation, and no-coverage-regression requirements.
Unsupported Core behavior must be described honestly, demonstrated by example-owned behavior where specified, or remain an explicit acceptance blocker.

**REQ-SCOPE-03 — Boundaries.** No Core API/protocol redesign, new NuGet packages, external services, web ports, automatic GitHub actions, or random-seed API is specified.
Do not add audit, provenance, signing-seal, or deployment subsystems. Child communication is local stdin/stdout only.
Future gated tests may acquire public assets; no weights, dependencies, or services are acquired by drafting this document.
The existing `StreamingTextGenerationExample` skeleton remains and must still be referenced in implementation documentation.

Future workflow, only after explicit authorization: fetch immediately before branching from fresh `development` into `feature/examples-suite`.
Repository guidance then requires verified GPG-signed commits, PRs targeting `development`, and monitoring authorized pushes to CI completion.
This document authorizes none of branching, committing, pushing, opening/merging PRs, dispatching workflows, publishing, or changing repository settings.

## 2. Verified baseline and plan corrections

**Verified facts** refer to inspected repository source; **Proposed** denotes the design to implement later, not an already-working feature.
Repository paths below are relative to the repository root unless used as Markdown links.

| Verified baseline | Specification consequence |
|---|---|
| `BaseInferenceEngine<TInput,TOutput>` owns model/session/provider/warmup/diagnostics lifecycle. | Reuse it; no copied session-management subsystem. |
| Existing `FaceDetectionEngine` uses registered `yolo11n.onnx`, 80 object labels, and SkiaSharp. | Detection is not face-only, despite the class name. |
| `TextEmbeddingEngine` is string → normalized `float[]`; its cosine helper is internal. | Implement a small Examples cosine helper, not a Core visibility change. |
| `BaseStreamingInferenceEngine` owns the loop, sampler, state disposal, and single-active-stream guard. | Implement its three hooks; full recompute is an example strategy, not KV-cache support proof. |
| Core declares `IncludePrefillInStream` and `ChunkSize` but does not implement their behavior in the loop or built-in Phi-3 engine. | Preserve both requirements through the new GPT-2 example's hooks/state; reject these flags in Phi-3 mode. |
| Base `PredictAsync` invokes streaming with default options and no cancellation token. | Configure example-owned defaults for one-shot; do not promise interruptible load/native execution. |
| Nonempty `ModelDownloadUrls` bypasses registry SHA-256 lookup in base model resolution. | An entry with a hash alone is insufficient; use the explicit verified provider design in section 7. |
| `TokenizerAssets` is internal and does not pass the option's byte cap to its downloader. | New GPT-2 acquisition uses public providers explicitly; built-in tokenizer traffic is not fully capped by the CLI option. |
| Worker load messages serialize only five engine-option fields. | Preacquire the model in the parent; never imply arbitrary provider/delegate/cap forwarding. |
| Worker metadata is not serialized; final `EngineInfo` and selected loading details are mirrored. | No invented proxy tensor metadata or full live worker-status feed. |
| Central `Microsoft.ML.Tokenizers` version is 2.0.1. | Verify tokenizer factories against that version; do not introduce a package/version change. |

The source plan's approximate 260 MB GPT-2 estimate is unverified; a candidate listing is approximately 329 MB.
Neither listing size nor a model-card/graph listing proves runtime compatibility, absence of mandatory past inputs, or self-contained weights.
Exact immutable model/tokenizer revisions, URL fallback pair, hashes, export metadata, and real execution remain unresolved gates.
Phi-3 bundle/tokenizer/graph compatibility is **UNKNOWN**, not a completed runnable demonstration.
The source plan's “repo-wide” coverage wording is a goal: actual reports must show new assemblies are measured.
The native-target duplication precedent is retained, but copying targets with best-effort commands is not proof of native/provider correctness.

## 3. Architecture and file layout

**REQ-ARCH-01 — Proposed structure.** Four thin console entry points call testable orchestration in `InferenceEngine.Examples`.
Each app references that library; no application contains preprocessing, model math, protocol framing, or provider-selection logic.
Use `OutputType=Exe`, `net10.0`, `IsPackable=false`, nullable/implicit usings, and the existing host-RID build pattern.
Keep engines/helpers in Examples, with XML documentation on public/protected APIs; Core remains unchanged except the planned registry data entry.

| Proposed artifact | Responsibility |
|---|---|
| `examples/DetectionExample/DetectionExample.csproj` and `Program.cs` | Dispatch to detection orchestration. |
| `examples/EmbeddingsExample/EmbeddingsExample.csproj` and `Program.cs` | Dispatch to semantic-search orchestration. |
| `examples/ChatExample/ChatExample.csproj` and `Program.cs` | Dispatch to sequential REPL/noninteractive generation. |
| `examples/WorkerExample/WorkerExample.csproj` and `Program.cs` | Early worker dispatch, otherwise proxy orchestration. |
| `InferenceEngine.Examples/Apps/` | App runners, parsed-settings records, option mapping, file/input/result seams. |
| `InferenceEngine.Examples/ConsoleDiagnostics.cs` | Shared status/session/check/fallback presentation through `ILogger`. |
| `InferenceEngine.Examples/CosineSimilarity.cs` | Small vector helper and stable corpus ranking. |
| `InferenceEngine.Examples/Gpt2StreamingGenerationEngine.cs` | BPE, full-recompute hooks, sampled-token buffering, default options. |
| `InferenceEngine.Examples/WorkerDetectionProxy.cs` | Typed detection proxy and stateless custom serializer. |
| `InferenceEngine.Examples/Apps/WorkerLaunchResolver.cs` | Apphost versus dotnet-entry-DLL launch description. |
| `InferenceEngine.Tests/*Example*Tests.cs` and engine-specific tests | Hermetic app/engine/helper tests and separately gated real paths. |
| `inference-engine-lib.slnx` | Add the four projects under `/examples/`. |
| `lockfiles/<AppName>.linux-x64.lock.json` | Four app lockfiles, plus affected existing graph updates. |

**REQ-ARCH-02 — Seams and ownership.** Inject engine factories, result `TextWriter`, diagnostic `ILogger`, REPL reader, asset resolver, and process-location information as needed.
Keep parsing/formatting/cosine/launch resolution pure where possible; do not expose a new public Core seam.
Examples may grant `InternalsVisibleTo` to `InferenceEngine.Tests` for tokenizer/output/state seams.
Tests can use Core's existing test-friend access to `WorkerFrame` and `SetWorkerStreamsForTesting`; production code cannot assume that internal access.
Use `using` for in-process engines and `await using` for proxies; dispose tensors, native outputs, tokenizer file streams, and disposable generation state on every exit.
Only one stream/request is active per interactive generation engine; the REPL does not add concurrent conversation scheduling.

## 4. Shared CLI and diagnostics contract

**REQ-CLI-01 — Proposed parsing.** Parse and validate before engine creation, native probes, network access, or cache-directory creation.
`--help` alone succeeds with usage; malformed combinations remain usage errors. Unknown flags and duplicate scalar flags are errors.
Repeated `--document` values are allowed; repeated `--worker-env` values require unique keys under the worker OS's environment-key comparison.
Parse numeric values invariantly; reject overflow, NaN/infinity, empty required values, and absent operands with actionable messages.
Normalize supplied file/directory paths without silently creating missing model/image/bundle inputs.

| Proposed common flag | Exact mapping/default | Applicability and validation |
|---|---|---|
| `--cpu` | `UseGpuAcceleration=false`; absent retains `true`. | All apps; requests CPU, not a special engine implementation. |
| `--device N` | `DeviceId=N`, default 0. | All apps; nonnegative Int32; provider/runtime determines availability. |
| `--model-path FILE` | Absolute graph file → `ModelPath` → `LocalModelProvider`. | All apps; existing nonblank file; incompatible with `--phi3 DIR`. |
| `--no-warmup` | `WarmupOnLoad=false`; absent retains `true`. | All apps; typed GPT-2 warmup is required. |
| `--max-download-mb N` | Checked `N * 1024L * 1024L` → `MaxModelDownloadBytes`; default 2 GiB. | All apps; positive integer MiB, not decimal MB; cap per acquired file, not total suite traffic. |
| `--mxr-cache DIR` | `MIGraphXModelCacheDirectory=DIR`. | All apps; absent null = default, explicit empty string = disabled; Linux-x64 MIGraphX only. |
| `--status` | Attach `OnStatusChanged`. | All apps; enables progress/status diagnostics, never result-stream pollution. |
| `--help` | Usage and exit 0. | All normal app modes; no engine/native/network work. |

GPU/device/cache settings are accepted but may be irrelevant under CPU selection or another provider; explain that rather than asserting acceleration.
A supplied `ModelPath` is a graph path, not a promise that the graph fits the engine's input/output schema.
Do not add a CLI that serializes an arbitrary `ModelProvider`; demonstrate provider injection through testable example construction.
A local path suppresses model acquisition only. MiniLM/Phi-3 tokenizer resolution can still access their cache/network.
For built-in tokenizer assets, explicitly disclose that `--max-download-mb` does not replace their current default downloader cap.
For new GPT-2 model/tokenizer acquisition and Worker parent acquisition, explicitly pass the selected cap to each public provider.

**REQ-CLI-02 — Proposed outcomes.** Exit 0 for successful output/help, 2 for usage/preflight input errors, 1 for acquisition/load/inference/protocol failure, 130 for observed cancellation.
Missing user-supplied input files/bundle members are usage errors; incompatible runtime graph/tokenizer behavior after preflight is a failure.
Cancellation is reported only when observed, not merely when Ctrl+C was pressed while an uncancellable operation remained in flight.
Catch at the app boundary to render an actionable failure and dispose owned resources; do not catch-and-skip actual inference failures.

**REQ-DIAG-01 — Proposed diagnostics.** Route logger diagnostics to stderr and results to injected writers; no `Console.WriteLine` logging.
Every app prints final `InferenceEngineInfo`: provider, platform/device, model name/path/source, fallback flag/reason, applied provider options, and ordered `Checks`.
Print each check's order/name/category/result/description and optional detail; retain the recorded evaluation order.
`--status` renders state, display-only `Detail`, partial failure `Info`, and `LoadError`; do not parse display text as a machine API.
Use `OnSessionInitialized` for success diagnostics and `OnStatusChanged` for progress/failure without duplicating the same final snapshot.
Guard shared writer/logger presentation against callback-thread interleaving; do not call engine methods reentrantly from callbacks.
Callback exceptions are contained by Core, but the helper itself must tolerate null partial information.
Display actual CPU fallback, not a synthetic “GPU success”; do not fabricate unpopulated `OnnxRuntimeVersion` or worker metadata.
Default result output is invariant: one detection/ranking record per line, or generated text chunks plus a final newline; prompts go to stderr.

## 5. DetectionExample

**REQ-DET-01 — Proposed behavior.** Require `--image FILE` plus applicable common flags; accept an existing nonempty image file.
Read bytes through an injected file seam, construct existing `FaceDetectionEngine`, call `LoadAsync`, then `PredictAsync`.
Use default registry model `yolo11n.onnx` unless a compatible local/provider override is supplied.
Print label, invariant confidence, and source-pixel `X`, `Y`, `Width`, `Height`; empty detections are a valid successful result.
Do not rename the library class or describe its COCO object detector as a face-only model.

**REQ-DET-02 — Existing pipeline demonstration.** Preserve `PreProcessWithContext` → `PostProcessWithContext` image-scale context passing.
Inspect metadata and `PrimaryInputName`; existing dynamic height/width fallback is 640, not a new configurable CLI image size.
Confidence 0.45 and IoU 0.5 are hardcoded engine thresholds; do not advertise threshold flags without separate approved engine changes.
CPU runs are the portable real-test baseline; GPU requests may fall back and must show why.
Hermetic app tests verify options/output with a fake engine; existing native Skia preprocessing tests are not mistaken for parser-only tests.
The gated `bus.jpg` test accepts at least one `person` OR `bus` OR `car`, not all three or exact boxes/counts.

## 6. EmbeddingsExample

**REQ-EMB-01 — Proposed behavior.** Require nonblank `--query TEXT`; accept repeated nonblank `--document TEXT`.
When no documents are supplied, use a small documented fixed corpus covering related and unrelated topics, with stable original indices.
Use built-in `TextEmbeddingEngine`, load once, embed the query and each document sequentially, then print rank, similarity, and document.
Do not hardcode 384 dimensions, add batching/truncation guarantees, or promise that an arbitrary replacement graph is MiniLM-compatible.
All common flags apply, subject to the built-in tokenizer-cap/local-path limitations in section 4.

**REQ-EMB-02 — Proposed cosine/ranking.** Keep the helper in Examples because Core's equivalent is internal.
Require equal-length, finite, nonempty vectors; mismatch/nonfinite/empty input yields a clear argument error in the helper.
Use double-precision accumulators for dot products/norms; either zero norm produces score 0 rather than NaN.
Rank by descending similarity, then original corpus index; identical scores preserve supplied order.
Hermetic tests cover identical/orthogonal/opposite/zero vectors, malformed input, stable ties, and formatting without native/network work.
A gated real-engine test requires `similarity(query,relevant) > similarity(query,unrelated)` rather than exact floating-point scores.

## 7. ChatExample and Gpt2StreamingGenerationEngine

### App modes and sampling

**REQ-CHAT-01 — Proposed modes.** `--prompt TEXT` runs one noninteractive request; otherwise read a sequential REPL until EOF or observed Ctrl+C.
REPL input is independent prompts with no hidden history, conversational service, instruction template, or persistence.
Default output is generated continuation only; GPT-2/distilgpt2 is not instruction-tuned and may produce irrelevant content.
`--one-shot` must actually call `PredictAsync` and demonstrate `BuildFinalOutput`, not manually concatenate a streaming app call.
Pass parsed generation settings as the new engine's configured `DefaultStreamingOptions` so one-shot flags remain effective.
Streaming calls use a fresh complete options object per request; passing per-call options replaces defaults rather than merging them.

| Proposed Chat flag | Mapping/default | Validation |
|---|---|---|
| `--max-steps N` | `StreamingOptions.MaxSteps`, default 256. | Positive Int32; counts all forward passes including prefill. |
| `--temperature X` | `Sampling.Temperature`, default 0. | Finite, nonnegative float; zero is greedy, positive is stochastic. |
| `--top-k N` | `Sampling.TopK`, default 0. | Nonnegative Int32; 0 disabled; at/above vocabulary size has no filtering effect. |
| `--top-p X` | `Sampling.TopP`, default 1. | Finite `0 < X <= 1`; 1 disabled. |
| `--include-prefill` | `IncludePrefillInStream=true`, default false. | New GPT-2 example only; exact original prompt prefix once. |
| `--chunk-size N` | `ChunkSize=N`, absent null behaves as 1. | Positive Int32; new GPT-2 example only. |
| `--eos-token IDS` | Comma-separated replacement `EosTokenIds`. | Unique nonnegative IDs, validated against verified tokenizer/logit vocabulary before inference. |
| `--phi3 DIR` | Select built-in Phi-3 local-bundle mode. | Existing directory; exclusive with `--model-path`; compatibility gate remains mandatory. |
| `--one-shot` | Invoke selected engine's `PredictAsync`. | Effective configured defaults; no promised token cancellation. |

Default GPT-2 EOS IDs must remain `[50256]` when constructing per-call options unless explicitly replaced.
Core's sampler is greedy for temperature <= 0; the CLI rejects negative values rather than inventing negative-temperature behavior.
Top-k filters by a kth-logit cutoff, so ties can retain more than k tokens; top-p retains a descending-probability nucleus.
Top-k/top-p only affect stochastic sampling; explain their inactivity at temperature 0 in help.
No deterministic random seed or exact stochastic sequence is promised. `RepetitionPenalty` is explicitly excluded and remains a follow-up, with no inert flag.

### Model and tokenizer acquisition

**REQ-CHAT-02 — Proposed verified acquisition.** Retain a `ModelName`, `ModelDownloadUrls` override demonstration, and a new `models.json` entry with pinned SHA-256.
When neither caller `ModelProvider` nor `ModelPath` is supplied, prepare default example options with an explicit `DownloadingModelProvider`.
Supply the verified registry SHA-256, matching override/registry URL list, cache directory, logger, and chosen byte cap to that provider.
Caller provider/path takes precedence; do not replace it during example default preparation or claim its validation guarantees.
This avoids the base hash trap: a nonempty URL override otherwise bypasses the registry hash entirely.
Because the provider is explicit, report actual `ModelSource.ExplicitPath`; do not relabel the diagnostic snapshot as registry-managed.
Keep registry lookup demonstrated by Detection and parent Worker acquisition; `models.json` must exist in every runnable app output.
Use public `DownloadingModelProvider` for GPT-2 tokenizer assets, with explicit byte caps and verified hashes, mirroring the internal `TokenizerAssets` pattern.
Exact model name, immutable revisions, primary/fallback URLs, model SHA-256, vocab/merges hashes and final size are **TBD pending verification**, not fabricated pins.
The fallback pair must identify identical verified bytes, not two incompatible exports or mutable-main substitutes.
Prove a single-file batch-1 full-recompute nonpast ONNX export, without required external weights or mandatory KV-cache inputs.
A listing/candidate description is not a successful ORT load; both metadata inspection and real inference are required before M3 acceptance.

**REQ-CHAT-03 — Proposed tokenizer.** Use centrally managed `Microsoft.ML.Tokenizers` 2.0.1 `BpeTokenizer`; do not guess a C# factory overload.
Configure `BpeOptions.ByteLevel=true`, no normalizer, a GPT-2-compatible case/whitespace-preserving regex pretokenizer, and `<|endoftext|>` ID 50256.
No automatic BOS/EOS insertion; sample-driven EOS remains engine-owned.
Verify the regex's contractions, optional-leading-space letters/numbers/symbols, and trailing-whitespace behavior against the selected immutable vocab/merges.
Preserve spaces, case, contractions, newlines, and Unicode bytes; model/tokenizer vocabulary must match.
Test real tokenizer vectors for those cases, plus fake-tokenizer seams; decoding a token independently is not a safe UTF-8 text-chunk strategy.
Use the official factory/options/pretokenizer references in section 14 and confirm their applicability to 2.0.1 before writing the engine.

### Tensor and output hooks

**REQ-CHAT-04 — Proposed full recompute.** Derive `Gpt2StreamingGenerationEngine` from `BaseStreamingInferenceEngine<string,string,string>`.
Implement `BuildStepInputs`, `ProcessStepOutputs`, and `BuildFinalOutput`; reduction is concatenation of emitted strings.
Step 0 tokenizes the prompt; later steps rebuild the prompt plus all previously sampled nonterminal IDs, batch size 1.
Bind declared `input_ids`, all-ones `attention_mask`, and sequential `position_ids`, using Int64/Int32 as verified graph metadata requires.
Reject incompatible required inputs/types/ranks clearly; do not silently zero-fill mandatory KV inputs or claim a KV-cache optimization.
Read float32 logits from the last sequence row; validate nonempty vocabulary/output shapes before `SampleNextToken`.
Validate prompt length and each growing sequence against the verified model context bound; never silently truncate.
Override typed warmup inputs, since Core's generic float dummy inputs are inadequate for integer token graphs.
Keep warmup independent of tokenizer/model download side effects beyond normal load; dispose all native inputs/outputs and disposable state.
Internal tokenizer/model-output seams support tensor-shape/type, logits, EOS, sampling, and reduction tests without real weights.

**REQ-CHAT-05 — Proposed text buffering.** New GPT-2 hooks/state, not Core's loop, implement both prefill and chunk controls.
`IncludePrefillInStream=true` emits the exact original prompt prefix once; false never emits that prefix.
Seed continuation decode state with the prompt's decoded boundary so prompt normalization/tokenization cannot leak prompt text into continuation output.
Count sampled non-EOS tokens toward `ChunkSize`; buffer text until that count reaches the configured size.
Decode accumulated token sequences with stable Unicode-prefix handling, not per-token UTF-8 fragments or replacement-character slicing.
EOS consumes a forward pass, returns `IsEos=true`, and is not rendered; the base still yields that hook's final string, possibly empty.
On EOS or the final allowed `MaxSteps` pass, flush the partial text buffer and any pending decode suffix exactly once.
Hooks know the current step and `CurrentOptions.MaxSteps`; do not add a Core end-of-stream hook to flush.
Core yields every hook step, so empty internal step strings may occur while buffering; the app ignores empty writes.
`ChunkSize` is a sampled-token buffering target, not a promise of exact characters per write or base-loop coalescing.
**Proof obligation:** tokenizer boundary stability, prompt stripping, EOS suppression, and final-pass flushing must be demonstrated together.
If this cannot be achieved with the current hooks, M3 remains blocked; do not drop knobs, lose final text, or redesign Core silently.
Stream cancellation may leave a partial continuation; release state/guard, do not invent a post-cancellation complete-output guarantee.

### Phi-3 and cancellation limitations

**REQ-CHAT-06 — Retained optional Phi-3 mode.** `--phi3 DIR` must demonstrate the built-in `TextGenerationEngine` from a compatible local bundle.
Expected graph is `phi3-mini-4k-instruct-cpu-int4-rtn-block-32.onnx`; verify that filename and every external-data reference against the selected official export.
Require the graph, its exact referenced external-data members (including the expected `.onnx.data`), and compatible `tokenizer.model` prerequisites.
Core resolves its tokenizer from the shared tokenizers cache/network, not automatically from the bundle directory.
Preflight must explain/prove compatible cache provisioning; a bundle-local tokenizer alone is insufficient under the current API.
Compatibility among built-in tokenizer construction, graph required inputs, and full-recompute execution is **UNKNOWN** and a mandatory real-run gate.
Keep this requirement open if incompatible; do not call it runnable, drop it, or fix Core without separate authorization.
Reject explicit `--include-prefill` and `--chunk-size` in Phi-3 mode because the built-in hooks ignore them.
For Phi-3 one-shot overrides, an Examples configured-default subclass may override the currently protected virtual `DefaultStreamingOptions`.
Retain Phi-3 default EOS `[32000,32002]`; test configured defaults and confirm that API seam before implementation.
If the seam no longer permits effective overrides, reject incompatible one-shot overrides with clear usage errors and report the limitation.
Do not claim the built-in output is continuation-only until prompt-boundary behavior is verified; label that unresolved difference in mode help.

**REQ-CHAT-07 — Cancellation.** Use a cancellation token for streaming and observe Ctrl+C between passes; unregister handlers in cleanup.
Core checks before a pass, not during native execution; downloads/load and one-shot lack the same token contract.
Do not simulate cancellation with an abandoned task while disposing an engine it still uses.
Test EOF, observed streaming cancellation, EOS, MaxSteps, failure, second-stream rejection, and reuse after stream cleanup.
Require one-shot/stream concatenation parity under identical configured defaults and greedy or otherwise controlled test outputs.

## 8. WorkerExample and WorkerDetectionProxy

### Acquisition and launch

**REQ-WORK-01 — Proposed parent.** Require `--image FILE`, applicable common flags, and the image-size guard below.
Before constructing/loading the proxy, resolve `yolo11n.onnx` using public registry entry URLs/SHA-256 and `DownloadingModelProvider`.
Pass the selected byte cap; skip acquisition when a validated local path/provider already resolves the model.
Forward only an absolute local model path. Parent acquisition logs are app diagnostics, not fabricated proxy load states.
Parent must not construct a detection engine, initialize ORT sessions, or invoke native capability probes.
Worker `ModelSource.ExplicitPath` is truthful even for parent-registry-acquired models; acquisition time is outside worker startup timeout.

**REQ-WORK-02 — Proposed launch.** Resolve apphost versus dotnet-entry-DLL using injected process/entry-assembly information.
For an apphost, launch the absolute current executable with `["--worker","inference"]`.
For `dotnet WorkerExample.dll`, launch the current dotnet host with `[absolute entry DLL,"--worker","inference"]`.
Use the entry assembly, not the shared Examples assembly; reject unavailable paths with an actionable error.
Use `ProcessStartInfo.ArgumentList`, never shell concatenation, and never recursively run `dotnet run`.
Route `--worker inference` before ordinary parsing, banners, help, logging setup that might write stdout, or host acquisition.
The dedicated child path constructs only `FaceDetectionEngine` and calls `InferenceWorkerHost.RunAsync<byte[],List<PredictionResult>>`.
Use a fixed descriptive `EngineId`; current host factory receives only options, so EngineId is not a multi-engine dispatch callback.
Normal mode prints detections through the same formatter as DetectionExample and owns the proxy with `await using`.

### Options, serialization, and stdout

**REQ-WORK-03 — Verified option boundary.** The load envelope forwards exactly `ModelPath`, `UseGpuAcceleration`, `DeviceId`, `WarmupOnLoad`, and `MIGraphXModelCacheDirectory`.
`ModelProvider`, `MaxModelDownloadBytes`, `OnStatusChanged`, and `OnSessionInitialized` are not serialized.
Attach callbacks in host options; the proxy mirrors final worker diagnostics and fires the host success callback.
Proxy metadata and primary input name remain null; do not display invented worker tensor information.
Set host warmup defaults explicitly; raw `WorkerEngineOptions` defaults differ from ordinary engine options.

| Proposed Worker-only flag | Contract |
|---|---|
| `--worker-env KEY=VALUE` | Repeated unique keys; nonempty key without `=`/NUL, value without NUL; empty value permitted. |
| `--startup-timeout-seconds N` | Integer 1–86400 seconds, default 180; maps to `StartupTimeout`. |
| `--inference-timeout-seconds N` | Integer 1–86400 seconds, default 60; maps to `InferenceTimeout`. |
| `--worker inference` | Internal child dispatch only; never treated as an ordinary parent request. |

Worker environment inherits parent values then applies overrides only in the child; host environment must remain unchanged.
Environment overrides are for nonsensitive tuning only: Core logs override values and worker arguments at information level.
Do not advertise secret-safe forwarding/redaction or print inherited environments.

**REQ-WORK-04 — Proposed custom serializer.** Override `CreateSerializer` with a stateless serializer; Core calls it during base construction.
Both endpoints use the same base64 image input and explicit JSON DTO output schema: Label, Confidence, X, Y, Width, Height.
Specify invariant numbers and reject null/malformed input/output; test empty detections and Unicode labels.
The default serializer already supports these DTOs; customization demonstrates the seam rather than claiming it is necessary.
Keep Core's envelope/framing unchanged: four-byte length-prefixed UTF-8 JSON, 16 MiB reader cap.
Reject images larger than 8 MiB before reading/encoding the whole request; test exact boundary and encoded envelope size.
The frame writer has no matching size guard; base64 and nested JSON overhead must remain under the reader cap.

**REQ-WORK-05 — Pipe safety.** Worker stdout contains protocol bytes only; no help, status, result, greeting, or logging text.
Configure child/engine logging to stderr before worker work; parent receives stderr through the existing logger forwarding.
Production host captures raw stdout and redirects managed `Console.Out` to stderr; native descriptor-1 writes are not automatically protected.
Require process-level stdout-purity tests with managed noise and real native loading; injected-stream tests alone cannot prove safety.

### Failure, status, and teardown

**REQ-WORK-06 — Verified limitations.** IPC is serialized; timeouts apply separately to frame write/read, not a total deadline including lock waiting/process spawn.
Timeouts kill the worker; load failures publish Failed, but predict failure/timeout need not change an already-Loaded snapshot.
No automatic crash recovery, successful-load restart, predict retry, instant cancellation, or full live worker-status stream is promised.
Proxy loading phases and best-effort MIGraphX compile-marker stderr details supplement final loaded/failure `EngineInfo`.
Proposed app behavior on fault is log, return failure, and dispose, not retry a possibly consumed prediction.
Prefer `await using`: best-effort shutdown frame, stdin-first pipe teardown, up to two seconds waiting for exit, then kill.
The shutdown write itself has no timeout; this is not a bounded-total-disposal or acknowledged-shutdown guarantee.
Test errors/EOF, blocked write/read timeouts, compile-marker status, concurrency, disposal, and actual child reaping.
Ctrl+C can stop subsequent work when observed; do not promise interruption of current tokenless proxy calls.

## 9. Feature-coverage matrix

Every original source-plan feature row is retained below, including explicit exceptions and unresolved gates.

| Library feature | Demonstration / requirement | Acceptance |
|---|---|---|
| `IInferenceEngine` contract (`LoadAsync`/`PredictAsync`) | All apps; Chat one-shot invokes the actual method. | AC-CLI-02, AC-CHAT-07 |
| `BaseInferenceEngine` extension (Pre/PostWithContext, context passing) | Existing Detection pipeline; streaming Chat inherits the lifecycle. | AC-DET-02, AC-CHAT-03 |
| `ModelName` / `ModelDownloadUrls` / `ModelRegistry` overrides | Chat URL override plus explicit verified provider; Detection registry lookup. | AC-CHAT-01, AC-DET-02 |
| `LocalModelProvider` via `ModelPath` | Common flag; Detection local graph path. | AC-CLI-02 |
| `DownloadingModelProvider` (cache, fallback URLs, sha256, `MaxModelDownloadBytes`) | Detection registry; explicit Chat/Worker providers; disclose built-in tokenizer cap gap. | AC-ASSET-01, AC-WORK-01 |
| `UseGpuAcceleration` / `DeviceId` | Common flags; Detection/Worker request and diagnostics. | AC-CLI-02, AC-BUILD-02 |
| `WarmupOnLoad` | Common `--no-warmup`; typed GPT-2 warmup. | AC-CLI-02, AC-CHAT-03 |
| `MIGraphXModelCacheDirectory` | `--mxr-cache`, null/empty/path distinction. | AC-CLI-02 |
| `OnStatusChanged` + `InferenceEngineStatus`/`Detail` | Shared status helper, including partial failure. | AC-DIAG-01 |
| `OnSessionInitialized` + `EngineInfo`/`Checks`/fallback reason | Shared final diagnostics, honest worker boundary. | AC-DIAG-01, AC-WORK-04 |
| Input/Output metadata inspection (dynamic shapes, `PrimaryInputName`) | Detection fallback; Chat schema validation; proxy intentionally lacks metadata. | AC-DET-02, AC-CHAT-03 |
| `BaseStreamingInferenceEngine` extension (3 hooks) | GPT-2 input/output/reduction hooks. | AC-CHAT-03, AC-CHAT-07 |
| `PredictStreamingAsync` + `StreamingOptions` (MaxSteps, EosTokenIds) | Chat complete per-call settings, pass bounds and sampled EOS. | AC-CHAT-04 |
| Sampling (Temperature/TopK/TopP) | Chat flags and crafted-logit tests. | AC-CHAT-05 |
| `IncludePrefillInStream` / `ChunkSize` | GPT-2 example-owned buffering; Phi-3 flags rejected. | AC-CHAT-06 |
| Cancellation at step boundaries | Chat streaming Ctrl+C, not native-run interruption. | AC-CHAT-08 |
| One-shot `PredictAsync` on a streaming engine (`BuildFinalOutput`) | Configured defaults and parity. | AC-CHAT-07 |
| `TextEmbeddingEngine` + `TokenizerAssets` | Semantic ranking, real tokenizer prerequisites. | AC-EMB-03 |
| `TextGenerationEngine` (built-in) | Retained local Phi-3 mode; compatibility remains blocking. | AC-PHI-01 |
| `OutOfProcessBaseInferenceEngine` + `InferenceWorkerHost` + protocol | Worker proxy and dedicated child path. | AC-WORK-03, AC-WORK-05 |
| `WorkerEnvironment` / timeouts / status mirroring | Nonsensitive overrides, separate timeouts, honest statuses. | AC-WORK-04 |
| `IWorkerSerializer` customization seam | Stateless matching base64/DTO serializer. | AC-WORK-03 |
| CPU fallback + ordered diagnostic trail | All apps' final diagnostics. | AC-DIAG-01, AC-BUILD-02 |
| Disposal (`IDisposable`/`IAsyncDisposable`) | In-process using; proxy await using; failure cleanup. | AC-ARCH-02, AC-WORK-04 |
| `SamplingParams.RepetitionPenalty` | Excluded: declared but not implemented; documented follow-up, no flag. | AC-DOC-01 |

## 10. Test and acceptance matrix

**REQ-TEST-01 — Layers.** Use xUnit, `Xunit.SkippableFact`, and `MethodName_Scenario_ExpectedResult` naming.
Hermetic parser/math/fake-engine/protocol tests need no model download, GPU, or real native session.
Native fixtures, tokenizer assets, and real subprocess tests are separate layers with explicit prerequisites and time budgets.
Use `FaceDetectionEngineTests` capability probes and `OutOfProcess*Tests` injected-stream/fault patterns without blanket exception-to-skip handling.
Skip only before execution for a concretely missing prerequisite; after prerequisites exist, wrong results, load failures, corrupt assets, and crashes fail.
Skipped real-model/GPU tests are not passed E2E evidence. Report skips and unexercised acceptance items separately.

| Acceptance ID | Required evidence | Milestone |
|---|---|---|
| AC-SCOPE-01 | Docs retain unapproved status/source plan; no source-plan or implementation changes in this task. | Draft / M0 |
| AC-ARCH-01 | Four nonpackable net10 executables, Examples orchestration, solution folder, Linux-x64 lockfiles. | M0 |
| AC-ARCH-02 | Fake runners verify ownership/cleanup after success, usage, failure, EOF/cancellation; XML docs and retained skeleton. | M0–M4 |
| AC-CLI-01 | Help/unknown/duplicate/missing operands, invariant parsing, overflow/NaN/infinity; no engine/native/network side effects. | M0–M4 |
| AC-CLI-02 | Every common flag maps exactly; file checks, byte cap, mxr null/empty/path, actual Load/Predict invocations. | M1–M4 |
| AC-DIAG-01 | Loaded/fallback/partial-failure snapshots, ordered checks, no duplicate final info, callback concurrency, stderr/result separation. | M0–M4 |
| AC-ASSET-01 | Hermetic HTTP/file seams prove cache hit, fallback, SHA mismatch, declared/streamed cap enforcement and partial-file cleanup. | M1–M4 |
| AC-DET-01 | Fake detections format labels/confidence/source coordinates; empty result and invalid image path behavior. | M1 |
| AC-DET-02 | Existing pipeline/dynamic-shape coverage plus gated local/downloaded YOLO bus image yields person OR bus OR car. | M1 |
| AC-EMB-01 | Cosine identical/orthogonal/opposite/zero, empty/mismatch/nonfinite, precision and stable ties. | M2 |
| AC-EMB-02 | Query/doc/default-corpus parsing, fake engine orchestration, ranking output and runtime dimension independence. | M2 |
| AC-EMB-03 | Gated MiniLM query scores relevant above unrelated; tokenizer/native prerequisites distinguished from real failures. | M2 |
| AC-CHAT-01 | Immutable nonpast single-file model + matching fallback/hash/provider precedence; no base-hash-bypass regression. | M3 gate |
| AC-CHAT-02 | Matched BPE tokenizer vectors for spaces/case/contractions/newlines/Unicode/special token and no implicit BOS/EOS. | M3 |
| AC-CHAT-03 | Fake outputs/metadata verify batch1 growing inputs, names/shapes/int types, masks/positions, float logits, bounds, typed warmup/disposal. | M3 |
| AC-CHAT-04 | Prefill counts in MaxSteps; custom/default EOS validation, EOS suppression and final-step handling; defaults not lost per call. | M3 |
| AC-CHAT-05 | Crafted logits verify greedy, temperature/top-k/top-p support and ties; no flaky exact stochastic text or seed API. | M3 |
| AC-CHAT-06 | Prefill false/true × chunk1/>1 × EOS/MaxSteps, Unicode boundary merges and final partial flush; no prompt leak/lost tail. | M3 |
| AC-CHAT-07 | One-shot calls PredictAsync with effective defaults; BuildFinalOutput equals streamed concatenation for controlled outputs. | M3 |
| AC-CHAT-08 | REPL sequential prompts, EOF, cancellation at boundaries, output failure, second-stream rejection and cleanup/reuse. | M3 |
| AC-CHAT-09 | Gated real GPT-2 CPU generation, MaxSteps=8, valid nonempty continuation; metadata/tokenizer listing alone insufficient. | M3 gate |
| AC-PHI-01 | Exact bundle/tokenizer prerequisites, rejected unsupported flags, effective one-shot defaults, real built-in generation and prompt behavior verified. | M3 gate |
| AC-WORK-01 | Parent registry/hash/cap/provider resolution forwards absolute path without loading ORT; honest model source and acquisition timing. | M4 |
| AC-WORK-02 | Pure apphost/dotnet-DLL resolver, missing-path errors, arguments with spaces, early child routing, no recursive dotnet run. | M4 |
| AC-WORK-03 | Serializer empty/Unicode/invalid/null round-trips, image8MiB boundary/envelope cap, five-field forwarding; injected host load/predict/shutdown. | M4 |
| AC-WORK-04 | Environment worker-only, timeout read/write faults, EOF/error/status/concurrency, no retry, disposal and child-reaping behavior. | M4 |
| AC-WORK-05 | Real apphost and dotnet-DLL self-spawn with spaced paths; stdout framing/log separation; gated actual worker detection. | M4 gate |
| AC-BUILD-01 | RID native cleanup/symlinks and actual ORT/provider selection validated per app output; project-reference target issue resolved. | M0 / M5 |
| AC-BUILD-02 | Three hosted OS CPU/fallback tests; separate capable Linux-x64 AMD run proves MIGraphX rather than accepting fallback. | M5 gate |
| AC-COV-01 | Comparable line/branch coverage baseline, all new assemblies/paths included, no regression, no invented numerical baseline. | Each / M5 |
| AC-DOC-01 | README app/base/domain/features matrix, examples guide, toc, CHANGELOG, skeleton link, limitations and follow-ups. | M5 |

Map requirement groups to acceptance: SCOPE→SCOPE; ARCH→ARCH/BUILD; CLI→CLI/ASSET; DIAG→DIAG.
DET→DET; EMB→EMB; CHAT→CHAT/PHI/ASSET; WORK→WORK/ASSET; TEST→all; BUILD→BUILD/COV/DOC; DONE→all.
For real worker processes use cross-platform built app binaries, not only the existing Linux `/bin/sh` fault harness.

## 11. Milestones M0–M5 and dependency order

All milestones are future implementation work, not actions performed or approved by this draft.
Each is independently reviewable/shippable only when its own acceptance gates pass; no feature-complete labels for placeholders.
Every milestone requires `./scripts/agent-self-check.sh`, focused tests, and comparable coverage with no regression.

| Milestone | Deliverables and dependency | Exit evidence |
|---|---|---|
| M0 — Scaffolding | After authorization/fresh development fetch: four projects/placeholders, solution folder, Linux-x64 locks, shared CLI/diagnostics seams and native target duplication. | ARCH/CLI/DIAG/BUILD/COV foundations; tooling configured and self-check green. |
| M1 — Detection | Depends M0; full existing-engine CLI, output/diagnostics and hermetic + gated image E2E. | DET/CLI/ASSET acceptance, bus labels and portable CPU path. |
| M2 — Embeddings | Depends M0; cosine helper, corpus/query ranking and built-in engine consumption. | EMB acceptance and gated ordering. |
| M3 — Chat | Depends M0 and resolved asset/API gates; GPT-2 hooks/tokenizer/registry entry, streaming/one-shot/REPL and retained Phi-3 mode. | CHAT/PHI acceptance; real GPT-2 and compatible local Phi-3 evidence mandatory. |
| M4 — Worker | Depends M0/M1; acquisition, proxy/serializer/launch, dedicated worker dispatch and full process tests. | WORK acceptance including both launch forms and actual isolated detection. |
| M5 — Docs/coverage/CI | Depends M1–M4; docs/toc/changelog, aggregate coverage, actual CI inclusion and platform/GPU validation. | DOC/COV/BUILD plus all preceding gates; only then an authorized PR workflow. |

M1 and M2 can proceed independently after M0; investigate M3's asset/Phi-3 prerequisites early to avoid late incompatibility surprises.
Do not postpone known hash, tokenizer, output-buffering, or native-target proof obligations to a documentation-only signoff.
Future signed commits/push/PR to `development` and CI monitoring require separate explicit authorization; merge/publish is not implicit in M5.

## 12. Build, coverage, CI, and documentation

**REQ-BUILD-01 — Native/build.** Duplicate `RemoveWindowsNativeLibsOnNonWindows` and `SymlinkLinuxNativeLibsForPInvoke` from Tests into each app output build.
Retain host-RID `RuntimeIdentifier`, `UseRidGraph`, `CopyLocalLockFileAssemblies`, and the six existing RID declarations.
Targets are output-local, not inherited from project references; preserve behavior while verifying quoting/paths with spaces and post-build file selection.
Linux symlinks are Unix-host/Linux-RID-specific; macOS is not a Linux-symlink test, and cross-compilation is not runtime proof.
Core's packaged `buildTransitive/InferenceEngine.Core.targets` is not automatically imported through a Core ProjectReference.
Prove the MIGraphX-capable ORT native selection in app output; explicitly reuse/import the existing target if needed, with no Core redesign.
Smoke-test loadability and provider behavior; `ContinueOnError=true` target success is insufficient.
Ensure `models.json` is copied to runnable outputs; keep package versions central and add no packages in this specification.

**REQ-BUILD-02 — Locks/platforms.** Generate four app Linux-x64 lockfiles and update affected existing locks after implementation approval.
`Directory.Build.props` names locks by host RID, even for an explicit different target RID; Linux-x64 locks cover that leg only.
Six declared RIDs are win-x64/win-arm64/linux-x64/linux-arm64/osx-x64/osx-arm64, not six validated runtime architectures.
Hosted build-test uses ubuntu-24.04, windows-2025, macos-15; validate their actual host RIDs without claiming every declared RID was executed.

**REQ-BUILD-03 — Future validation commands.** Run in the repository only after implementation/tooling approval; no commands below were executed for this draft.
Use `./scripts/agent-self-check.sh` per milestone; its current tests exclude FaceDetectionEngineTests and NativeLibraryValidationTests and collect no coverage.
Run focused `dotnet test InferenceEngine.Tests/InferenceEngine.Tests.csproj --filter "FullyQualifiedName~<AreaTests>"` for actual new test-class names.
For CI parity use locked restore, Release no-restore build, and Release no-build tests with `--collect:"XPlat Code Coverage"`.
Also run `dotnet test --collect:"XPlat Code Coverage"` with documented comparable filters/capabilities; explicitly execute gated real-model/process paths.
No install/restore/build/test/model acquisition is part of authoring this specification.

**REQ-BUILD-04 — Coverage.** Compare actual line and branch reports before/after under the same OS/RID/configuration/filter/prerequisite conditions.
Include new Examples and thin-app assemblies; prove collection rather than assuming every repository file is measured.
Current ordinary Tests uses Coverlet; Core.IntegrationTests does not declare its collector. No runsettings-based app inclusion was found.
No numerical baseline was measured here. Restore or exceed the comparable baseline; do not hide untested logic with exclusions.
`codecov.yml` target95%/tolerance3 percentage points and informational patch status do not relax AGENTS' no-regression requirement.

**REQ-BUILD-05 — CI.** Preserve `build-test.yml` three-OS restore/build/test/coverage and `rocm-gpu-test.yml` capable self-hosted GPU validation.
Actual build-test excludes only NativeLibraryValidationTests; the composite publish action excludes FaceDetection tests as well.
GPU workflow currently has no PR trigger, omits `examples/**` push paths, and filters only existing FaceDetection/RocmProvider classes.
Future approved CI edits must include relevant app/engine/helper paths and new gated GPU classes/filters; choose trusted trigger policy explicitly.
Do not assume a feature PR executes GPU tests: job additionally requires `ROCM_RUNNER_ENABLED=true` and a registered AMD runner.
GPU prerequisites include Linux x64, accessible `/dev/kfd`, HSA/HIP/MIGraphX libraries and provider native assets.
Hosted runs skip GPU-specific tests cleanly; a capable GPU run must assert MIGraphX, not silently pass on CPU fallback.
Coverage upload/comment jobs are best-effort and not equivalent to passing measured gates; do not require new services or ports.
`publish-nuget.yml` also triggers on development/main; no publishing workflow is dispatched by this task or implicitly by the spec.

**REQ-BUILD-06 — Future documentation.** M5 adds README Examples app/base/domain/features matrix, `examples/README.md`, `docs/examples-guide.md`, `docs/toc.yml` entry, and CHANGELOG.
Include commands only after the corresponding apps exist; mark prerequisites, first-download costs, flags per mode, cache locations, offline/tokenizer limits, and failure examples.
Retain skeleton/streaming guide links, explicit repetition-penalty follow-up, Phi-3 compatibility status, and GPU-validation limitations.
The present README change is only a planned/unimplemented specification link, not a runnable-suite announcement.

## 13. Risks, prerequisites, and definition of done

| Gate | Current status / required resolution |
|---|---|
| GPT-2 asset selection | Unresolved: actual self-contained nonpast graph, immutable identical-byte fallback pair, size/context/schema, registry and tokenizer hashes. |
| Tokenizer API/configuration | Unresolved: confirm 2.0.1 factories/pretokenizer settings and matched real vectors; do not infer compatibility from newer docs alone. |
| Buffered text semantics | Unresolved implementation proof: exact prompt prefix, continuation boundary, Unicode merges, EOS and final-MaxSteps partial flush with current hooks. |
| Built-in Phi-3 path | UNKNOWN: local graph/external data/tokenizer provisioning, required inputs, one-shot defaults and output prompt behavior must be verified by a real run. |
| Native app outputs | Unresolved: duplicated targets plus project-reference ROCm native selection and real apphost/DLL smoke evidence. |
| Validation tooling | Read-only probes found dotnet, trunk and pwsh absent; `.trunk/` also absent despite guidance. Bash/Python are available. |
| Coverage baseline | Not measured; new assembly inclusion and comparable line/branch baseline must be established during implementation. |
| GPU execution | Runner enablement/availability and expanded workflow inclusion must be verified; current workflow configuration does not prove execution. |

These are future implementation/setup prerequisites, not blockers to drafting this document.
No builds, tests, restores, coverage collection, dependency installations, model downloads, or external GitHub writes were performed for this specification.
No claim of a passed runtime, three-OS, or GPU gate follows from documentation/link/whitespace checks.

**REQ-DONE-01 — Definition of done after approval.** All four apps and required new engines/helpers exist, are documented, and satisfy mapped acceptance IDs.
Every original feature-matrix row has demonstrated evidence or its explicit declared exclusion; Phi-3 and GPT-2 proof obligations cannot be silently reclassified as exclusions.
All hermetic tests pass; required real-model/process/platform/GPU gates have actual results, with skipped paths recorded as unverified.
No comparable line/branch coverage regression; new paths and assemblies are measured; native outputs and resource cleanup are verified.
Per-milestone self-check and focused/full coverage checks pass, public/protected documentation exists, and docs/toc/changelog are complete.
Only separately authorized repository operations proceed, with signed commits and successful required CI; unapproved merge/release/publish actions remain out of scope.

## 14. Source and research references

These references explain existing behavior or guide future verification; their presence is not proof of downloaded assets or working implementations.
Future artifact paths in the layout/documentation sections are intentionally not links to nonexistent files.

| Reference | Relevance |
|---|---|
| [Source plan](examples-suite-plan.md) and [AGENTS](../AGENTS.md) | Scope, milestone/feature preservation, signed workflow, tests/coverage rules. |
| [Base engine](../InferenceEngine.Core/BaseInferenceEngine.cs) and [options](../InferenceEngine.Core/InferenceEngineOptions.cs) | Provider precedence/hash trap, lifecycle, callbacks, metadata. |
| [Streaming base](../InferenceEngine.Core/BaseStreamingInferenceEngine.cs) and [streaming options](../InferenceEngine.Core/StreamingOptions.cs) | Hooks, sampling, tokenless one-shot, declared-but-unimplemented controls. |
| [Embedding engine](../InferenceEngine.Core/Text/TextEmbeddingEngine.cs), [generation engine](../InferenceEngine.Core/Text/TextGenerationEngine.cs), [tokenizer assets](../InferenceEngine.Core/Text/TokenizerAssets.cs) | Built-ins, tokenizer access/cap limitations, Phi-3 assumptions. |
| [Detection engine](../InferenceEngine.Examples/FaceDetectionEngine.cs) and [skeleton](../InferenceEngine.Examples/StreamingTextGenerationExample.cs) | Existing vision pipeline and retained educational streaming skeleton. |
| [Out-of-process proxy](../InferenceEngine.Core/OutOfProcessBaseInferenceEngine.cs) and [options](../InferenceEngine.Core/OutOfProcessEngineOptions.cs) | Self-spawn, timeouts, logging, status, shutdown limits. |
| [Worker host](../InferenceEngine.Core/Worker/InferenceWorkerHost.cs), [protocol](../InferenceEngine.Core/Worker/WorkerProtocol.cs), [serializer](../InferenceEngine.Core/Worker/WorkerSerializer.cs) | Dispatch/option subset, framing size and customization. |
| [Registry](../InferenceEngine.Core/ModelProviders/ModelRegistry.cs), [downloader](../InferenceEngine.Core/ModelProviders/DownloadingModelProvider.cs), [models](../InferenceEngine.Core/models.json) | Public acquisition, fallback/cache/hash/cap, YOLO entry. |
| [Diagnostics](../InferenceEngine.Core/Diagnostics/InferenceEngineInfo.cs) and [status](../InferenceEngine.Core/Diagnostics/InferenceEngineStatus.cs) | Actual snapshots, ordered checks, display-only details. |
| [Detection tests](../InferenceEngine.Tests/FaceDetectionEngineTests.cs), [IPC faults](../InferenceEngine.Tests/OutOfProcessIpcFaultTests.cs), [IPC behavior](../InferenceEngine.Tests/OutOfProcessEngineTests.cs), [concurrency](../InferenceEngine.Tests/OutOfProcessConcurrencyTests.cs) | Capability gates, fake streams, Linux process harness, status/cleanup conventions. |
| [Tests project](../InferenceEngine.Tests/InferenceEngine.Tests.csproj), [Core project](../InferenceEngine.Core/InferenceEngine.Core.csproj), [Examples project](../InferenceEngine.Examples/InferenceEngine.Examples.csproj), [native target](../InferenceEngine.Core/buildTransitive/InferenceEngine.Core.targets) | Output-local native handling and consumer-versus-project-reference distinction. |
| [Build properties](../Directory.Build.props), [central packages](../Directory.Packages.props), [solution](../inference-engine-lib.slnx) | Host-RID locks, Tokenizers 2.0.1, existing solution membership. |
| [Self-check](../scripts/agent-self-check.sh), [build-test](../.github/workflows/build-test.yml), [composite action](../.github/actions/build-and-test/action.yml), [GPU workflow](../.github/workflows/rocm-gpu-test.yml), [coverage policy](../codecov.yml) | Actual filters/triggers, tooling and coverage limitations. |
| [BpeTokenizer.Create API](https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.tokenizers.bpetokenizer.create?view=ml-dotnet) | Official factory reference; confirm exact 2.0.1 applicability. |
| [BpeOptions source](https://raw.githubusercontent.com/dotnet/machinelearning/v5.0.1/src/Microsoft.ML.Tokenizers/Model/BpeOptions.cs) and [RegexPreTokenizer source](https://raw.githubusercontent.com/dotnet/machinelearning/v5.0.1/src/Microsoft.ML.Tokenizers/PreTokenizer/RegexPreTokenizer.cs) | Byte-level options and regex pretokenization; source tag is not a substitute for package-version verification. |
| [Candidate distilgpt2 ONNX listing](https://huggingface.co/Xenova/distilgpt2/tree/main/onnx) | Discovery candidate only; mutable listing, approximate size, no accepted model pin/runtime proof. |
| [Official Microsoft Phi-3 cpu-int4 bundle metadata](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32) | Verify exact graph/external-data/tokenizer bundle and immutable revision before accepting the optional mode. |
