# Plan: Ship InferenceEngine.Core 1.1, validate MIGraphX GPU, implement Section-B diagnostics in InferenceApp

Three workstreams across two repos + one system task. **Dependency:** Section-B consumes the new `1.1` package, so publish 1.1 first (Workstream 1). Workstream 2 (ROCm/GPU) is independent and can run in parallel.

---

## Workstream 1 — lib: `VERSION_PREFIX` → `1.1` + publish `1.1.0-dev.N`
Repo: `intel-agency/inference-engine-lib`. No source edit — this is a repo variable + a workflow run. (The csproj default is already `1.1.0-dev`; only the `VERSION_PREFIX` repo variable is stale at `1.0`.)

- **1.1** Set the repo variable: `gh variable set VERSION_PREFIX --body 1.1` (repo admin).
- **1.2** Publish a new dev build. `publish.yml` has `workflow_dispatch`: `gh workflow run "Publish NuGet" --ref development`, then `gh run watch <id> --exit-status`. Expect computed version `1.1.0-dev.<run>`.
- **1.3** Confirm on the feed: `gh api orgs/intel-agency/packages/nuget/InferenceEngine.Core/versions` lists `1.1.0-dev.N`.
- Out of scope: stable `1.1.N` + NuGet.org (needs release promotion); the dev package is enough for testing.

---

## Workstream 2 — lib: install ROCm runtimes + validate MIGraphX GPU on this Debian/AMD box
**System task — the `apt`/`sudo` steps cannot be run by a code agent; the user runs them.** Current state: `/dev/kfd` present, AMD RX 6700 XT, but only the ABI-incompatible `libamdhip64.so.5` (HIP 5.7); `libamdhip64.so.7` + `libmigraphx_c.so.3` are **missing** → MIGraphX currently falls back to CPU.

- **2.1** Add the AMD ROCm apt repo (one-time) — see `docs/inferenceapp-migraphx-migration-guide.md` §7.
- **2.2** `sudo apt install hip-runtime-amd migraphx` (NOT Debian's `.so.5`; NOT the heavy `rocm`/`rocm-libs` meta).
- **2.3** Verify: `find /usr/lib -name 'libamdhip64.so.7'` and `libmigraphx_c.so.3` exist; `rocm-smi` works. **Do NOT** symlink `.so.5`→`.so.7` (ABI-incompatible; the multi-SONAME HIP probe was discarded for exactly this reason).
- **2.4** GPU validation via the lib's SkippableFact `EngineInfo_LinuxX64WithKfd_MIGraphXWithAllChecksPassed` (`InferenceEngine.Tests/EngineInfoTests.cs`). After install it must PASS with `Provider == MIGraphX` (it currently skips/falls back). Run in a shell with `NUGET_AUTH_TOKEN` exported (restore needs it).

---

## Workstream 3 — InferenceApp: implement full Section B (B.1–B.5)
Repo: `nam20485/InferenceApp`, branch `dev/performaer-id-adaptation`. Stack: Avalonia 12.0.4 + CommunityToolkit.Mvvm 8.4.2.

**Pre-req:** Workstream 1 done. Bump both refs `InferenceApp.UI.csproj:22` and `Tests/Tests.csproj:21` from `1.0.0-dev.11` → `1.1.0-dev.N`. `dotnet restore`/`build` with `NUGET_AUTH_TOKEN` exported.

**Grounding (verified):**
- `OnnxInferenceEngine : BaseInferenceEngine<byte[], List<PredictionResult>>` and `ArcFaceRecognitionEngine : BaseInferenceEngine<byte[], float[]>` → both already expose `.EngineInfo`.
- Engines are DI singletons in `App.axaml.cs` ConfigureServices (~lines 70–90), each via `new InferenceEngineOptions()` with **no `OnSessionInitialized` yet**.
- `MainViewModel` already has `[ObservableProperty] private string engineStatusText` (line 37) — reuse for the footer.
- CLI subcommand pattern: `Program.cs:14` (`if (args[0]=="import-performers")`).

**B.1 — `EngineDiagnosticsViewModel` (new, CommunityToolkit.Mvvm):** `ViewModels/EngineDiagnosticsViewModel.cs`, `partial : ObservableObject`. Properties: `DetectionInfo`/`RecognitionInfo` (`InferenceEngineInfo?`), `DetectionChecks`/`RecognitionChecks` (`IReadOnlyList<ProviderDiagnosticCheck>`), and formatted `DetectionSummary`/`RecognitionSummary` (`"{Provider}/{DeviceId}"` or `"CPU (fallback: {FallbackReason})"`).

**B.2 — UI placement:**
- Footer in `MainWindow.axaml`: compact `[Detect: {summary}] [Embed: {summary}]`, color cue on `IsFallback` (DataTrigger/value-converter). Bind to `EngineDiagnosticsViewModel` (reuse `MainViewModel.engineStatusText` if convenient).
- Diagnostics section in `SettingsWindow.axaml` (note: file is `SettingsWindow.axaml`, NOT `SettingsView.axaml` as the v2 doc says): per-engine full `EngineInfo` + ordered `Checks` via an `ItemsControl` with ✓/✗/— glyphs + each `Detail`.

**B.3 — `--diag inference` CLI (`Program.cs`):** sibling to `import-performers`. `if (args.Length>1 && args[0]=="--diag" && args[1]=="inference")`. Load both engines with `WarmupOnLoad=false`, print provider/platform/device/fallback + the full ordered checks trail; exit 0 on success, non-zero if `LoadAsync` throws on every path.

**B.4 — Consumer tests (`Tests/`):** add `Xunit.SkippableFact` to `Tests.csproj` if absent. Add `OnnxInferenceEngine_LoadAsync_PopulatesEngineInfo` (Provider != Unknown, Checks non-empty, `Checks[i].Order == i`); optionally a `LazyOnnxInferenceEngine` reload-refresh test.

**B.5 — Untouched:** keep the `import-performers` `libonnxruntime_providers_rocm.so` strip step as-is (deliberate, per the v2 doc).

**Wiring:** In `App.axaml.cs` ConfigureServices, add `OnSessionInitialized = info => { /* marshal to UI thread; push into EngineDiagnosticsViewModel */ }` to **both** `InferenceEngineOptions` instances (detection + recognition); register `EngineDiagnosticsViewModel` as a DI singleton. The callback runs on a thread-pool thread — marshal to the Avalonia UI thread before mutating bindable state.

---

## Risks / edge cases
- **Token:** both repos' restores hang unless `NUGET_AUTH_TOKEN` is exported in the shell; persist in `~/.bashrc` so VS Code DevKit's background `--interactive` restores also resolve.
- **GPU timing:** until Workstream 2 lands, Section-B will legitimately show `CPU` + a granular `FallbackReason` (e.g. `MIGraphXRuntimeMissing`) — that *is* the diagnostics working correctly.
- **LazyOnnxInferenceEngine** wraps the concrete engine via the UI `IInferenceEngine` contract; confirm it exposes the inner `.EngineInfo` (or read `EngineInfo` from the concrete engine the DI holds), since `IInferenceEngine` doesn't surface it.
- **ORT version:** keep NO direct `Microsoft.ML.OnnxRuntime` reference in InferenceApp — it resolves 1.24.1 transitively; adding a mismatched direct ref breaks native ABI.
- **Avalonia binding:** the Checks list needs an `ItemsControl` + a converter/DataTrigger for the ✓/✗/— glyph and the `IsFallback` color cue.

## Validation (definition of done)
- `1.1.0-dev.N` published; InferenceApp restores it; `dotnet build` green.
- `EngineInfo_LinuxX64WithKfd_MIGraphXWithAllChecksPassed` PASSES with `Provider == MIGraphX` after ROCm install.
- InferenceApp `--diag inference` prints the real provider/trail; footer + SettingsWindow show diagnostics; consumer tests pass.
- InferenceApp config/csproj changes committed on `dev/performaer-id-adaptation`.

## Out of scope
- Stable release promotion (`development → main → release`) / NuGet.org.
- Installing ROCm runtimes is a user/sudo operation (documented, not agent-executed).
