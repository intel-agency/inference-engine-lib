# Analysis: Out-of-Process IPC Encoding & Throughput (`OutOfProcessBaseInferenceEngine`)

**Status:** Analysis / recommendation (handoff for the InferenceEngine.Core team)
**Date:** 2026-07-10 (revised)
**Target:** `InferenceEngine.Core` — `OutOfProcessBaseInferenceEngine<TInput,TOutput>` and `Worker/*`
**Related:** [`out-of-process-inference-engine-design.md`](out-of-process-inference-engine-design.md), [`consumer-integration-guide.md`](consumer-integration-guide.md), [`streaming-inference-guide.md`](streaming-inference-guide.md)

> **Revision note (2026-07-10).** An earlier draft recommended Protocol Buffers as the binary
> encoding. On review that is the **wrong tool for this layer**: protobuf has no generics, so it
> cannot express the library's `TInput`/`TOutput`. This revision recommends a **generic-friendly
> binary codec (MessagePack)** for the binary win, and demotes protobuf/gRPC to an optional,
> *concrete-types-only* future path. See §4–§5.

---

## TL;DR

The out-of-process (OOP) inference engine marshals `Load`/`Predict` between host and a child worker
over a **length-prefixed JSON pipe** (redirected `stdin`/`stdout`). For the workload it was designed
for — **single-shot image inference** — JSON is **not** the bottleneck; GPU inference dominates and
the pipe's bandwidth is effectively unbounded.

For **streaming video** (detection + identification at frame rate, which the consumer app also supports)
video still does **not** need more bandwidth (a pipe moves GB/s); what it needs is a **cheaper
encoding** to drop the per-frame base64 (+33%) and double-JSON tax.

**Recommendation:** replace the JSON encoding with a **simple binary stream — MessagePack** — over
the existing pipe, behind the existing `IWorkerSerializer<TInput,TOutput>` seam. One phase. Then stop.

- MessagePack is **generic-friendly** (`Serialize<T>` works with arbitrary `TInput`/`TOutput`) and
  binary, so `byte[]` flows raw (no base64) and there is no double-JSON-encode. No transport change,
  no schema/codegen.
- For the consumer app's video case it is **sufficient**: inference dominates the frame budget, so the
  one-in-flight round-trip fits comfortably at 30 fps once the encoding tax is removed.
- **MemoryPack** is the faster alternative but requires consumer type annotation, so it cannot be the
  transparent library default — only a consumer opt-in. See §6.
- **Protocol Buffers / gRPC** are *not* recommended at the library layer. Protobuf has no generics,
  and the host↔worker path is the same .NET binary, so protobuf's schema/cross-language dividends do
  not apply. Reserve gRPC for a *possible* later phase **only if** profiling shows the serialized
  round-trip (not the GPU) is the video limiter — and there it uses *concrete* messages, not the
  generic seam. See §9.

---

## 1. Current channel characteristics (grounded in the code)

### 1.1 Transport — an OS anonymous pipe, not a socket

The host and worker communicate through redirected child-process stdio:

- `OutOfProcessBaseInferenceEngine.cs:174-200` — `ProcessStartInfo { UseShellExecute = false,
  RedirectStandardInput = true, RedirectStandardOutput = true }`, then
  `_toWorker = _process.StandardInput.BaseStream` and `_fromWorker = _process.StandardOutput.BaseStream`.
- On Linux these are `pipe(2)` file descriptors; on Windows, anonymous pipes. They are **purely
  local and kernel-mediated** (memcpy-class), not named pipes, Unix domain sockets, shared memory,
  or network sockets.

Implication: the transport has GB/s-class throughput and sub-microsecond handoff. It is **never**
the throughput bottleneck for any realistic inference payload.

### 1.2 Framing — 4-byte length prefix + UTF-8 JSON

`WorkerProtocol.cs:49-113` (`WorkerFrame`): every message is `<4-byte little-endian length><UTF-8
JSON body>`, hard-capped at **16 MiB** (`MaxFrameBytes`, `WorkerProtocol.cs:54`) as a guard against
a corrupt length prefix forcing a giant contiguous allocation. Frame buffers are rented from
`ArrayPool<byte>` rather than allocated per frame.

### 1.3 Encoding — System.Text.Json, with two real inflation sources

1. **`byte[]` input → base64** (`WorkerSerializer.cs:24`, STJ's built-in behavior). Base64 inflates
   a binary image by **+33%** (4 characters per 3 bytes).
2. **Double JSON-encoding of the payload.** `IWorkerSerializer.SerializeInput/Output` returns a
   `string` (`WorkerSerializer.cs:11-17`), which is stored in `WorkerEnvelope.Payload`
   (`WorkerProtocol.cs:37`) and then the **entire envelope is serialized to JSON again**
   (`InferenceWorkerHost.cs:137-141`). So the inner object's quotes/backslashes are re-escaped,
   roughly doubling the payload's contribution to the wire size.

### 1.4 Concurrency — strictly one request in flight

`OutOfProcessBaseInferenceEngine.cs:41` — `_inFlight = new SemaphoreSlim(1, 1)`. All host↔worker IPC
is serialized so concurrent `PredictAsync` calls cannot interleave write/read pairs on the shared
pipe (the "H1" guard from `review-fix-plan.md`). There is **no pipelining and no multiplexing**;
throughput is bounded by `1 / RTT`, where one round trip is:

```
host encode → write → worker decode → preprocess → INFER → postprocess → worker encode → write → host decode
```

Timeouts: 3 min startup (first MIGraphX compile ≈ 85 s), 60 s per predict
(`OutOfProcessEngineOptions.cs:34-37`). Redirected process pipes do not reliably honor
`CancellationToken`, hence the `WaitAsync(timeout)` + `KillWorker()` pattern
(`OutOfProcessBaseInferenceEngine.cs:210-248`).

---

## 2. Why JSON is fine for single-shot image inference

For the designed workload (`FaceDetectionEngine`: `byte[]` → `List<PredictionResult>`,
`FaceDetectionEngine.cs:18`), a typical frame round trip breaks down roughly:

| Stage | Size | Approx. cost |
|---|---|---|
| Input image (JPEG/decoded) | ~100 KB–2 MB on wire (×1.33 base64) | STJ base64 ser+deser ≈ 1–3 ms |
| Output (handful of boxes, `FaceDetectionEngine.cs:204`) | ~1–3 KB | negligible |
| Pipe transfer (both ways, ~2 MB) | — | < 2 ms (pipes run multiple GB/s) |
| **MIGraphX GPU inference** | — | **~10–30 ms steady-state** (one-time ~85 s compile) |

JSON ser/deser of a ~1 MB image is **single-digit milliseconds**, while GPU inference is ~10–30 ms
and dominates the ~15 ms RTT. The pipe's bandwidth is not exercised. **The encoding is not the
bottleneck**, and the serialized round-trip leaves ample slack inside the 60 s timeout. No change is
warranted for single-shot use. This matches the design doc's own position
(`out-of-process-inference-engine-design.md:267`): *"IPC cost per inference is negligible for
image/single-shot inference."*

---

## 3. What changes for streaming video

The consumer app also does streaming-video face recognition and identification. Video does **not** need
more bandwidth (the pipe handles GB/s). What it stresses, in priority order:

### 3.1 Continuous JSON+base64 CPU (the part the encoding fixes)

The severity depends on what is sent over the wire:

- **Compressed frames (JPEG).** `FaceDetectionEngine` takes compressed `byte[]` and runs
  `SKBitmap.Decode` itself, so frames are small (~2–6 MB/s at 30 fps). Continuous JSON+base64
  ser/deser is a modest few-%-of-a-core cost. Noticeable but not fatal.
- **Raw decoded RGB.** 1920×1080×3 ≈ 6.2 MB/frame → ~186 MB/s at 30 fps → ~248 MB/s after base64.
  Continuous JSON tokenization + base64 encode *and* decode becomes ~0.5 GB/s of pure CPU
  serialization work — **a full core or more**, competing directly with the Avalonia render thread
  and the inference worker.

Binary encoding removes this tax entirely (`byte[]` flows as raw bytes).

### 3.2 The one-in-flight serialization (the part the encoding does NOT fix)

With `SemaphoreSlim(1,1)` the per-frame cost is strictly serial (see §1.4). For ~15–30 ms GPU
inference at 30 fps (33 ms/frame budget), the slack is thin, and — critically — **frame N+1's
transfer and preprocessing cannot overlap frame N's inference**. The design doc flags exactly this:
*"For high-FPS video, design the worker to stay hot and accept a frame stream rather than one
process-round-trip per frame"* (`out-of-process-inference-engine-design.md:267`).

This is an **architectural** gap, not an encoding gap. **Changing the encoding alone does not fix
it** — any codec over the current pipe still serializes every frame behind the semaphore. This is why
§8 (video fit) hinges on whether video is *inference-bound* or *IPC-bound*.

### 3.3 Output direction is negligible

Per-frame detections are a few hundred bytes to a few KB; even double-JSON-escaped they are trivial
relative to the input frame. The encoding win is almost entirely on the **input** path.

---

## 4. The decisive constraint: protobuf has no generics (why it is the wrong tool here)

`OutOfProcessBaseInferenceEngine<TInput,TOutput>` is generic; `PredictionResult` lives in the
consumer (`InferenceEngine.Examples/FaceDetectionEngine.cs:204`), not in the library. The library's
serialization seam is generic:

```csharp
// WorkerSerializer.cs:11
public interface IWorkerSerializer<TInput, TOutput>
{
    string SerializeInput(TInput input);
    TInput DeserializeInput(string serialized);
    string SerializeOutput(TOutput output);
    TOutput DeserializeOutput(string serialized);
}
```

**Protocol Buffers cannot back this seam.** Proto3 has no generics — every message is a concrete,
compile-time-known type. There is no `Serialize<T>`; there is `Foo.Parser.Parse(...)` for one concrete
`Foo`. The ways people try to reconcile protobuf with generics all fail at this layer:

1. **Opaque `bytes` payload.** Hides `TInput`/`TOutput` as `bytes` — but then protobuf is just framing
   a byte stream it never interprets, and the actual typing happens in a *separate* serializer behind
   `bytes`. Protobuf's "strong typing" dividend never touches the payload (the only data that
   matters).
2. **`Google.Protobuf.WellKnownTypes.Any` / `Struct` / `Value`.** `Any` still needs a concrete
   registered type to deserialize; `Struct`/`Value` are JSON-in-protobuf (you lose the binary
   efficiency you came for). Heavyweight, ugly, still not generic.
3. **Give up generics.** Define concrete `.proto` messages per consumer type — throws away
   `OutOfProcessBaseInferenceEngine<TInput,TOutput>` and `IWorkerSerializer<TInput,TOutput>`, which
   exist precisely because the library **cannot** own the consumer's types.

**Conclusion:** protobuf and generic type arguments are incompatible at the same layer. The host↔worker
path is also the **same .NET binary** (the consumer app launched with `--worker`), so protobuf's
cross-language and schema-evolution dividends — its real strengths — do not apply here either. This
is why protobuf's dividends diminish for this use case, and why the recommendation changed.

---

## 5. Codec choice for a generic seam

What the binary win actually requires is a codec with a genuine **`Serialize<T>`** that works with
arbitrary generic type params, treats `byte[]` as raw bytes (no base64), and needs no schema/codegen
to start:

| Codec | Generic `Serialize<T>` | `byte[]` raw | Unannotated POCOs | Schema/codegen | Speed | Cross-lang |
|---|---|---|---|---|---|---|
| **System.Text.Json** (today) | yes | **no** (base64 +33%) | yes | none | medium | via JSON |
| **MessagePack-CSharp** | yes | yes (`bin`) | **yes** (contractless mode) | none to start | fast | yes |
| **MemoryPack** | yes | yes | **no** (`[MemoryPackable]` required) | source-gen required | **fastest** | .NET only |
| **Protocol Buffers** | **no generics** | yes | no (concrete messages only) | `.proto` + codegen | fast | yes |

**MessagePack** is the clear fit for the **library default**: generic-friendly, binary, handles
arbitrary consumer POCOs without forcing annotations, and `byte[]` flows raw. It slots directly into
`IWorkerSerializer<TInput,TOutput>` with no contract/transport change.

**MemoryPack** is faster but **cannot be the transparent library default** (§6) — it is a consumer
opt-in for max speed.

**Protobuf** is ruled out at this layer by §4.

---

## 6. Would MemoryPack work for this codebase?

MemoryPack (Cysharp) is the fastest .NET serializer — for blittable types it writes structs directly
to memory (`Span<byte>`), near-memcpy. It is **generic-friendly**
(`MemoryPackSerializer.Serialize<T>(value)` / `Deserialize<T>(bytes)`), so at first glance it looks
ideal for the generic seam. But it has one hard constraint:

> **MemoryPack requires non-primitive types to be decorated `[MemoryPackable]` (source-generated).**
> Primitives like `byte[]`, `int`, `string` serialize without annotation; a POCO like
> `PredictionResult` **must** be `[MemoryPackable] partial` or MemoryPack throws at runtime.

Applied to this codebase (`OutOfProcessBaseInferenceEngine<byte[], List<PredictionResult>>`,
`FaceDetectionEngine.cs:18`):

- **`TInput = byte[]`** → serializes as raw `bin`, **no annotation needed.** ✔
- **`TOutput = List<PredictionResult>`** → `PredictionResult` (`FaceDetectionEngine.cs:204`,
  `string Label` + 5 floats) **must be annotated `[MemoryPackable]`**. ✘ for the library default.

Because the library does not own `PredictionResult`, it **cannot** require that annotation. So:

- MemoryPack **cannot be the transparent library-default serializer** (unlike MessagePack, whose
  contractless mode serializes unannotated POCOs).
- MemoryPack **is** an excellent **consumer opt-in**: the consumer app annotates `PredictionResult` once
  and selects a `MemoryPackWorkerSerializer<TInput,TOutput>` behind `CreateSerializer()`
  (`OutOfProcessBaseInferenceEngine.cs:58`) for maximum speed.

Other MemoryPack notes relevant here:

- **Format is MemoryPack's own, not MessagePack-compatible** (the name is misleading). .NET-only.
  Fine here — host and worker are the same .NET consumer binary.
- **Version tolerance** assumes both ends share the same type definition and serializer version; more
  brittle than MessagePack/protobuf. Acceptable for "same binary, same version" (this repo's
  host↔worker case); risky if host and worker are ever revved independently.

**Recommendation:** ship a MessagePack default; optionally also ship a `MemoryPackWorkerSerializer`
for consumers who annotate their types and want the extra speed. For `byte[]`→POCO engines the win
is on the small `TOutput`, so MessagePack's speed is already ample and MemoryPack is a micro-optimization.

---

## 7. Recommended change — MessagePack binary stream over the existing pipe

**Keep** the transport (stdio pipe) and the 4-byte length-prefix framing (`WorkerFrame`). **Swap**
the JSON body for MessagePack, and make the serializer seam byte-oriented.

### 7.1 Make `IWorkerSerializer` byte-oriented (small breaking change)

Change the contract from `string` to `byte[]` / `ReadOnlySpan<byte>` so binary payloads flow natively:

```csharp
public interface IWorkerSerializer<TInput, TOutput>
{
    byte[] SerializeInput(TInput input);
    TInput DeserializeInput(ReadOnlySpan<byte> serialized);
    byte[] SerializeOutput(TOutput output);
    TOutput DeserializeOutput(ReadOnlySpan<byte> serialized);
}
```

- **`SystemTextJsonWorkerSerializer`** becomes `JsonSerializer.SerializeToUtf8Bytes(...)` / `Deserialize<T>(bytes)`
  — already UTF-8 bytes; now it is no longer base64-wrapped-in-an-envelope-string, so even the default
  gets slightly better.
- Add **`MessagePackWorkerSerializer<TInput,TOutput>`** using `MessagePackSerializer.Serialize<T>(...)`
  (contractless mode handles arbitrary POCOs; `byte[]` → raw `bin`).

This is a **breaking change** to the `public IWorkerSerializer`/`SystemTextJsonWorkerSerializer`
surface, but it is internal to the OOP/Worker IPC and the change is well-motivated. Existing consumers
that never touch the interface are unaffected; the only implementors are the library defaults.

### 7.2 Swap `WorkerFrame` to MessagePack bodies

Framing becomes `<4-byte LE length><MessagePack body>`. The `MaxFrameBytes` guard and `ArrayPool`
buffering in `WorkerFrame` (`WorkerProtocol.cs:49-113`) are retained unchanged; only
`JsonSerializer.Serialize/Deserialize` calls become MessagePack calls. Nesting the payload as a
MessagePack `bin` is essentially free (length prefix, no escaping), so the double-encode tax of the
JSON path disappears.

### 7.3 What this buys

- **`byte[]` input flows raw** — no +33% base64.
- **No double-encode** of the payload (binary nesting, not string re-escaping).
- **Faster** per-frame ser/deser than STJ (~2–5× for objects; ~memcpy for `byte[]`).
- **No transport change, no schema/codegen** — a new `MessagePack` package reference only.

### 7.4 What this does NOT buy (be explicit)

It does **not** raise the one-in-flight ceiling: the `SemaphoreSlim(1,1)` serialization
(`OutOfProcessBaseInferenceEngine.cs:41`) is unchanged, so frame overlap is still impossible. Whether
that matters for video is the next section.

---

## 8. How MessagePack-over-pipe fits the streaming-video use case

The decisive question is whether video is **inference-bound** or **IPC-bound**.

**What MessagePack-over-pipe fixes for video:**

- `byte[]` input → raw `bin`, no base64, no continuous base64 encode/decode CPU.
- No double-JSON-encode; faster per-frame ser/deser.
- For the **raw-RGB worst case** (~186 MB/s at 1080p/30 fps): MessagePack makes `byte[]` essentially
  memcpy, so the per-frame serialization CPU collapses from "a full core" to near-nothing. Large win
  if raw frames are ever sent.
- For the **compressed-frame case** (JPEG, the realistic consumer app path): the IPC CPU tax was
  already modest; MessagePack removes it entirely.

**What it does not fix:** the `SemaphoreSlim(1,1)` serialization (§3.2). Frames still cannot overlap.

**Verdict for the consumer app (MIGraphX face detection):**

- Inference ≈ 10–30 ms/frame dominates. With MessagePack, `inference + ipc_rtt` ≈ 15–30 ms + ~1–2 ms,
  comfortably **inside the 33 ms budget for 30 fps**. The one-in-flight model has ample slack, and the
  encoding tax is gone. **MessagePack-over-pipe is sufficient** for 30 fps video.
- It would only become insufficient if inference dropped to a few ms **and** target fps rose to 60+
  (making IPC serialization the limiter), or if frame *overlap* were needed to keep the GPU saturated
  — which only helps when the GPU has idle gaps between frames (it often does not when saturated).

**Conclusion:** "convert JSON → MessagePack and call it good" is a **defensible stopping point** for
the consumer app's video workload. It removes the per-frame encoding tax, and the serialized round-trip
fits inside the frame budget because inference dominates. The pipelining/overlap work (§9) is only
warranted if future profiling shows IPC — not the GPU — is the limiter.

---

## 9. Optional future — gRPC, only if pipelining is proven necessary

If, after MessagePack, profiling shows the serialized round-trip (not the GPU) is the video limiter,
the next lever is **protocol shape**, not encoding: a **streaming/pipelined** channel so frame N+1
transfer/preprocess overlaps frame N inference. At that point gRPC over a Unix domain socket is the
natural choice:

```proto
service InferenceWorker {
  rpc PredictStream(stream PredictRequest) returns (stream PredictResponse);
}
```

This gives a continuous frame stream, multiplexing/overlap, deadlines, and flow-control.

**Important caveats — why this is optional and deferred:**

- gRPC requires **concrete** protobuf messages (`PredictRequest`/`PredictResponse`) — this is
  deliberately *outside* the generic `OutOfProcessBaseInferenceEngine<TInput,TOutput>` seam (§4). It
  would be a separate, type-specific streaming service, not a replacement for the generic engine.
- gRPC adds an HTTP/2 server, client stubs, and startup cost — **overkill** while the single-shot /
  one-in-flight path is sufficient.
- The MessagePack binary work (§7) is **not wasted** if this later happens: the framing, the
  `byte[]`-oriented seam, and the per-frame CPU reductions all carry forward. The only addition is
  the streaming transport and concrete request/response messages.

**Do not start here.** Land MessagePack (§7), measure real video fps, and only revisit if IPC — not
inference — is the proven bottleneck.

---

## 10. Decision summary

| Workload | Bottleneck today | Action |
|---|---|---|
| Single-shot image | GPU inference (~10–30 ms); JSON IPC ≈ single-digit ms | **None required.** JSON is acceptable. |
| Streaming video — encoding/CPU | base64 (+33%) + double-JSON + continuous ser/deser CPU | **§7:** MessagePack binary stream over the existing pipe (one phase). |
| Streaming video — overlap/throughput | `SemaphoreSlim(1,1)` prevents frame overlap | **None unless profiling proves it.** Optional §9 (gRPC streaming) only if IPC, not GPU, is the limiter. |

**Net:** the channel is latency-bound by serialized GPU inference, not by its text encoding or the
pipe. The encoding win is real but modest, and the right tool is a **generic-friendly binary codec
(MessagePack)** — not protobuf, which has no generics and whose strengths (schema, cross-language)
do not apply to a same-binary host↔worker path. MemoryPack is a consumer opt-in for extra speed if
types are annotated, but cannot be the transparent library default. Convert to MessagePack over the
existing pipe, measure video fps, and stop there unless profiling says otherwise.

---

## 11. Code references

| Concern | Location |
|---|---|
| Host proxy, semaphore, pipe wiring | `InferenceEngine.Core/OutOfProcessBaseInferenceEngine.cs:41,58,174-200,210-248` |
| Serializer seam (overridable; the contract to make byte-oriented) | `OutOfProcessBaseInferenceEngine.cs:58` (`CreateSerializer`); `Worker/WorkerSerializer.cs:11-17` |
| STJ default impl (base64 for `byte[]`) | `InferenceEngine.Core/Worker/WorkerSerializer.cs:24-42` |
| Length-prefixed framing, 16 MiB cap, `ArrayPool` | `InferenceEngine.Core/Worker/WorkerProtocol.cs:49-113,54` |
| `WorkerEnvelope` (body to swap JSON→MessagePack) | `InferenceEngine.Core/Worker/WorkerProtocol.cs:27-47` |
| Double-JSON payload encode (host & worker) | `OutOfProcessBaseInferenceEngine.cs:133,144`; `InferenceWorkerHost.cs:137-141` |
| Worker message loop (request/response, one in flight) | `InferenceEngine.Core/Worker/InferenceWorkerHost.cs:49-84` |
| OOP options / timeouts | `InferenceEngine.Core/OutOfProcessEngineOptions.cs:34-37` |
| Example consumer types (`byte[]`→`List<PredictionResult>`) | `InferenceEngine.Examples/FaceDetectionEngine.cs:18,204` |
| Design doc — video caveat | `docs/out-of-process-inference-engine-design.md:267` |
