# Evaluation: .NET Standard 2.0 Compatibility for `InferenceEngine.Core`

**Status:** Recommendation ready (decision pending owner approval)
**Scope:** `InferenceEngine.Core` only (Tests + Examples intentionally excluded — see §6)
**Current target:** `net10.0`, C# 13, nullable enabled
**Question answered:** Is it feasible and beneficial to make `InferenceEngine.Core` consumable by legacy runtimes via a **netstandard2.0** target?

---

## 1. TL;DR / Verdict

| Option | Feasible | Recommended |
|---|---|---|
| **.NET Framework 2.0 (net20, 2005)** | ❌ No | No — impossible |
| **.NET Standard 2.1** | ✅ (technically) | No — excludes .NET Framework entirely, defeats the goal |
| **netstandard2.0 multi-target** (`net10.0;netstandard2.0`) | ✅ **Yes** | ⚠️ **Conditional** — only if a concrete legacy consumer exists |
| **net48 (Framework 4.8) multi-target** | ✅ Yes | Inferior to netstandard2.0 for reach |

**Bottom line:** A netstandard2.0 target is **feasible and verifiable** — the hard dependency (ONNX Runtime 1.24.1) confirmed ships a `netstandard2.0` assembly. The work is **moderate** (multi-target + ~5 polyfill packages + conditional compilation for ~6 modern APIs). The **benefit is real but modest and shrinking**: the only audience that gains anything is **legacy .NET Framework 4.6.1+ Windows desktop apps** doing CPU/DirectML inference. **Recommendation: do not invest unless you have a named consumer on .NET Framework.** If you do, proceed via the spike in §9.

---

## 2. Clarifying the ".NET 2.0" ambiguity (resolved)

"Downgrade to .NET 2.0 / .NET Fx compatible" conflates three unrelated targets. They are NOT interchangeable:

- **.NET Framework 2.0** (`net20`) — released **2005**. Predates async/await (2012), LINQ (2007), ONNX Runtime, `Task<T>`, `Span<T>`, records, nullable refs. This library cannot exist there. Ruled out.
- **.NET Standard 2.0** (`netstandard2.0`) — released **2017**. The standard "broad compatibility" library baseline. Consumed by .NET Framework **4.6.1+**, .NET Core 2.0+, .NET 5/6/7/8/9/10, Mono, Xamarin. **This is what "make it compatible" means in practice.** ← evaluated here.
- **.NET Standard 2.1** — supported by .NET Core 3+ and .NET 5+, but **NOT by any .NET Framework version**. Choosing 2.1 to get a few extra BCL APIs would **eliminate** the exact legacy-Framework audience you're trying to reach. Do not use 2.1 if Fx compatibility is the goal.

Owner decision (captured this session): **evaluate netstandard2.0.**

---

## 3. Current state of `InferenceEngine.Core`

- `InferenceEngine.Core.csproj`: `<TargetFramework>net10.0</TargetFramework>`, `<ImplicitUsings>enable`, `<Nullable>enable`, `<WarningLevel>9999>`.
- Build is deeply coupled to modern SDK/runtime features:
  - `UseRidGraph`, six `RuntimeIdentifiers`, host-RID auto-detection via `$([RuntimeInformation]::RuntimeIdentifier)`.
  - Post-build MSBuild targets: `RemoveWindowsNativeLibsOnNonWindows`, `SymlinkLinuxNativeLibsForPInvoke`, `ForceRocmOnnxRuntimeLibOnLinuxStandalone` — all written for .NET 10's P/Invoke resolution quirks.
- API surface spans: `BaseInferenceEngine`, `BaseStreamingInferenceEngine`, `OutOfProcessBaseInferenceEngine`, `ModelProviders/*`, `Diagnostics/*`, `Text/*` (tokenizer engines), `Worker/*` (out-of-process IPC).

---

## 4. Dependency feasibility — VERIFIED from local NuGet cache

| Package | Version | netstandard2.0 target? | Evidence |
|---|---|---|---|
| `Microsoft.ML.OnnxRuntime.Managed` | 1.24.1 | ✅ **Yes** | `lib/net8.0`, `lib/netstandard2.0` present in cache; nuspec `System.Memory` dep |
| `Microsoft.ML.OnnxRuntime` (meta) | 1.24.1 | ✅ Yes | delegates to `.Managed` |
| `Microsoft.ML.Tokenizers` | 1.0.3 | ✅ Yes | `lib/net8.0`, `lib/netstandard2.0` in cache |
| `Microsoft.Extensions.Logging.Abstractions` | 10.0.3 | ✅ Yes | ships netstandard2.0 (well-known) |
| `System.Memory` | (transitive) | ✅ Yes | provides `Span<T>` on netstandard2.0 |
| `Microsoft.ML.OnnxRuntime.DirectML` | 1.23.0 | ✅ Windows-only; works on netstandard2.0/Fx | DirectML EP, Windows RID only |
| `InferenceEngine.ROCm.Runtime.linux-x64` | 1.24.1.36 | n/a | native-only package, irrelevant to TFM |
| `Microsoft.SourceLink.GitHub` | 8.0.0 | ✅ dev-only | build-time, no runtime impact |

**Conclusion: no dependency blocks netstandard2.0.** The ORT managed binding itself was the make-or-break item and it passes.

---

## 5. API gap analysis — the project's OWN code (this is the real work)

The library compiles for net10.0. To also compile for netstandard2.0, these modern APIs need a polyfill or `#if` rewrite. **None require changing net10.0 behavior** — all changes are additive/conditional.

### 5a. Solvable with NuGet polyfill packages (low effort)

| API used | Where | netstandard2.0 fix |
|---|---|---|
| `Span<T>`, `ReadOnlySpan<T>`, `Memory<T>`, `stackalloc` span | `BaseStreamingInferenceEngine.cs:281` (`NthLargest`), tokenizers | Already transitively available via ORT's `System.Memory` dep; pin explicitly |
| `ArrayPool<T>.Shared` | `BaseStreamingInferenceEngine.cs:173,209,287` | Add `System.Buffers` package |
| `IAsyncEnumerable<T>`, `await foreach`, `[EnumeratorCancellation]` | `BaseStreamingInferenceEngine.cs:55` (whole streaming engine) | Add `Microsoft.Bcl.AsyncInterfaces` package |
| `IAsyncDisposable`, `ValueTask` | `OutOfProcessBaseInferenceEngine.cs:34,273` | `Microsoft.Bcl.AsyncInterfaces` + `System.Threading.Tasks.Extensions` |
| `ImmutableDictionary` | `BaseInferenceEngine.cs:599,603` | Add `System.Collections.Immutable` package |
| C# language features (records, file-scoped ns, switch expr, pattern matching, nullable refs, target-typed `new`, `using var`) | everywhere | `<LangVersion>latest</LangVersion>` + add `PolySharp` for compiler-required attributes (`IsExternalInit`, `Nullable`, etc.) |

### 5b. Requires conditional compilation `#if !NETSTANDARD2_0` / code rewrite (moderate effort)

| API used | Where | netstandard2.0 status | Fix |
|---|---|---|---|
| `NativeLibrary.TryLoad` | `BaseInferenceEngine.cs:322` (GPU dlopen probes: libhsa, libamdhip64.so.7, libmigraphx_c.so.3) | ❌ .NET Core 3+ only — no Fx equivalent | Rewrite behind an abstraction; on netstandard2.0 P/Invoke `dlopen`/`LoadLibrary` directly. **Largest single item.** |
| `Task.WaitAsync(TimeSpan)` | `OutOfProcessBaseInferenceEngine.cs:221,241` | ❌ .NET 6+ only | `#if NET6_0_OR_GREATER` … `#else` a `Task.WhenAny(t, Task.Delay(timeout))` fallback (note: small timer allocation; acceptable for IPC path) |
| `ProcessStartInfo.ArgumentList` | `OutOfProcessBaseInferenceEngine.cs:183` | ❌ .NET 5+ only | `#if` … `#else` build a quoted `Arguments` string for netstandard2.0 |
| `psi.Environment[...]` indexer | `OutOfProcessBaseInferenceEngine.cs:185` | ⚠️ present on .NET Core; on .NET Framework use `.EnvironmentVariables[...]` | `#if` branch |
| `MemoryExtensions.Sort` (span sort) | `BaseStreamingInferenceEngine.cs:214` (`order.AsSpan().Sort(...)`) | ❌ Not in `System.Memory` package | `#if NET5_0_OR_GREATER` … `#else` manual sort (e.g., `Array.Sort` on a copied int[], or an insertion sort for top-k) |
| `Span.Fill`, `Span.BinarySearch`, `Span.CopyTo` | `BaseStreamingInferenceEngine.cs:290,299,349` | ⚠️ `CopyTo` ok via System.Memory; `Fill`/`BinarySearch` span extensions came in .NET Core 2.1+ **not** in the System.Memory NuGet | `#if` fallbacks: loop-fill, manual binary search |
| `Random.Shared` | `BaseStreamingInferenceEngine.cs:241` | ❌ .NET 6+ only | Replace with a `static readonly Random` field |
| `Nameof`/string interpolation, LINQ, `HttpClient`, `SemaphoreSlim.WaitAsync`, `Lazy<T>`, `Path.Combine` | various | ✅ All present in netstandard2.0 | No change |

### 5c. Build/MSBuild assets that need conditional gating

These net10-specific pieces must be disabled or guarded for the netstandard2.0 build:
- `UseRidGraph`, `RuntimeIdentifiers`, host-RID `RuntimeIdentifier` property-function.
- All three post-build `<Target>`s (`RemoveWindowsNativeLibsOnNonWindows`, `SymlinkLinuxNativeLibsForPInvoke`, `ForceRocmOnnxRuntimeLibOnLinuxStandalone`) — gate with `Condition="'$(TargetFramework)' == 'net10.0'"` (or `!= 'netstandard2.0'`).
- The Linux native-lib symlink strategy exists to satisfy **.NET 10's** P/Invoke search algorithm. On .NET Framework the resolution path differs; a netstandard2.0 consumer on Fx is responsible for placing/finding the native lib. Document this; do not port the symlinks.

---

## 6. What stays net10.0-only

- **`InferenceEngine.Tests`** — uses `Xunit.SkippableFact`, code coverage, integration tests downloading models. Keep on net10.0. No benefit to a netstandard2.0 test assembly.
- **`InferenceEngine.Examples` (`FaceDetectionEngine`)** — depends on `SkiaSharp`. Could be multi-targeted but adds SkiaSharp/Fx matrix complexity for a "reference" project. Defer unless a consumer asks.
- **Streaming + out-of-process engines** — *can* be netstandard2.0 (via `Microsoft.Bcl.AsyncInterfaces`) but a .NET Framework consumer must also reference that package and gains `IAsyncEnumerable` through it. Consider gating these features to net10.0 via `#if` and exposing only the one-shot `LoadAsync`/`PredictAsync` contract on netstandard2.0 to shrink the surface. (Design decision — see §8 Q1.)

---

## 7. Benefit assessment (skeptical)

**Who actually gains from a netstandard2.0 target?**
1. **Legacy .NET Framework 4.6.1+ Windows desktop apps** (WPF/WinForms) not yet migrated to .NET 8/10 — can do CPU and DirectML inference. This is a **real but shrinking** audience in 2026.
2. **Maintenance-only / locked-down systems** that cannot move off .NET Framework.

**Who gains nothing:**
- New projects in 2026 are on .NET 8/10 — they consume the **net10.0** target directly today. netstandard2.0 adds nothing for them.
- Linux MIGraphX/ROCm acceleration is Linux + modern runtime only regardless of TFM.
- CoreML (macOS) similarly modern-runtime oriented.

**Honest assessment:** The library's value proposition is **GPU-accelerated ONNX inference with a modern diagnostics surface**. That surface (`NativeLibrary` probes, RID graphs, async streaming) is intrinsically a modern-runtime story. Porting to netstandard2.0 reaches an audience whose apps are, by definition, not where ONNX inference is most useful. **Benefit = low-to-moderate and declining over time.**

---

## 8. Risks & open questions

1. **Streaming/out-of-process scope on netstandard2.0 (design decision):** Expose full surface (more polyfill surface, consumer must also reference `Microsoft.Bcl.AsyncInterfaces`) **or** gate streaming/OOP to net10.0 and ship one-shot-only on netstandard2.0? → *Recommended: gate to net10.0 to minimize risk and consumer friction.*
2. **.NET Framework native-lib loading:** A Fx consumer must correctly deploy the matching `onnxruntime.dll` + DirectML native binaries. This is a packaging/docs burden, not solved by the TFM.
3. **Test coverage gap:** netstandard2.0 build gets no automated test coverage unless a net48 test project is added (cost). Otherwise the netstandard2.0 target is "compiles-clean" only.
4. **`System.Text.Json`** (used by `Worker/` serializers): available for netstandard2.0 via the NuGet package, works on Fx 4.7.2+ with known perf/feature caveats. Only relevant if OOP is exposed on netstandard2.0.
5. **`Microsoft.ML.OnnxRuntime` 1.24.1 is near end-of-line** (cache shows 1.26/1.27 exist). A future bump could drop the netstandard2.0 target and force re-evaluation. Re-verify on every ORT upgrade.

---

## 9. Recommendation & gated path forward

**Recommendation: Do NOT multi-target unless a specific, named .NET Framework consumer exists.** The cost/benefit doesn't justify it speculatively.

**If a concrete consumer exists, proceed in this order (spike-first):**

1. **Spike — prove compilation (gate).** Temporarily add `netstandard2.0` to `<TargetFrameworks>`; add polyfill packages (`System.Memory`, `System.Buffers`, `Microsoft.Bcl.AsyncInterfaces`, `System.Collections.Immutable`, `System.Threading.Tasks.Extensions`, `PolySharp`); run `dotnet build -f netstandard2.0`. Capture the full compiler-error list — that list IS the §5b work manifest.
2. **Decision gate after spike:** if error list is ≈ the §5b table (expected), continue; if ORT exposes members missing on netstandard2.0 (unexpected), stop and reassess.
3. Add `#if` conditionals for §5b items, prioritizing `NativeLibrary.TryLoad` abstraction (re-enable GPU-probe path; Fx uses `LoadLibrary`/`dlopen` P/Invoke).
4. Gate net10-only MSBuild targets (§5c) on `$(TargetFramework)`.
5. (Optional) gate streaming + OOP engines to net10.0 (§8 Q1).
6. Verify: `dotnet build` both TFMs; run `dotnet test` on net10.0 (netstandard2.0 gets compile-only verification unless a net48 test project is added).
7. Decide packaging: one nupkg with both TFMs (preferred) — confirm `buildTransitive` `.targets` only applies to net10.0 RID consumers.
8. Document Fx consumer setup (native lib deployment, required `Microsoft.Bcl.AsyncInterfaces` if streaming exposed).

**Estimated scope:** ~5 polyfill packages, ~6 conditional-compilation sites, 1 new abstraction for native loading, build-target gating. Non-trivial but bounded **if** the spike confirms the §5b error set.

---

## 10. If the original intent really was ".NET Framework 2.0"

For completeness: net20 (2005) is **infeasible** — not "hard", *impossible* for this library. It would require dropping ONNX Runtime, async/await, LINQ, `Span`, nullable refs, records, and rebuilding the entire library against a runtime that lacks the target platform's hardware stacks. There is zero benefit and total loss of the library's purpose. Do not pursue.

---

*Evidence basis: TFM tables verified against the local NuGet package cache (`~/.nuget/packages`) on 2026-07-08; API-gap list derived from reading `BaseInferenceEngine.cs`, `BaseStreamingInferenceEngine.cs`, `OutOfProcessBaseInferenceEngine.cs`, `InferenceEngine.Core.csproj`, and the `Diagnostics`/`ModelProviders` sources.*
