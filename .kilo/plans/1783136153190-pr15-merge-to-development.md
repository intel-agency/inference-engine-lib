# Plan: Safe-commit the Section-B + ROCm-doc work (two repos)

**Mode note:** Authored in Plan Mode — execution (validation + commits) is done by an
implementation-capable agent. This plan IS the commit authorization artifact.

## Phase 1 — Secret scan (DONE)
- **Lib repo:** clean (exit 0). 3 files in scope, no sensitive content.
- **InferenceApp repo:** exit 1, but **all 3 findings are false positives** — confirmed:
  - `SettingsViewModel.cs:49` `_qdrantApiKey = settings.QdrantApiKey;` — pre-existing
    config-passing (runtime `AppSettings` value, no literal secret); not a line this change added.
  - `SettingsViewModel.cs:147` `QdrantApiKey = defaults.QdrantApiKey;` — pre-existing reset line.
  - `VideoViewModel.cs:131` `cancellationToken: _processCts.Token` — regex noise (`cancellationToken`).
  - None introduce a literal secret; none are added lines. **InferenceApp commit requires explicit
    user authorization to proceed past the non-zero scan exit** (safe-commit rule). The lib commits
    are unblocked (scan clean).

## Phase 2 — Validation (implementer runs before each commit)
- **Lib:** changes are doc-only (`docs/` + memory bank); build/test unaffected. Optionally
  `dotnet build` to confirm no source accidentally touched. CI gate: `./scripts/agent-self-check.sh`.
- **InferenceApp:** `dotnet test` (builds `InferenceApp.UI` + `Tests`). Last local run: **57/57 pass**.

## Phase 3–4 — Grouping (3 commits across 2 repos)

| # | Repo | Commit message | Files |
|---|---|---|---|
| 1 | lib | `docs: rewrite linux-rocm-config.md for the MIGraphX execution provider` | `docs/linux-rocm-config.md` |
| 2 | lib | `docs: refresh memory bank context with current state` | `.kilocode/rules/memory-bank/context.md` |
| 3 | InferenceApp | `feat: expose Engine-Info diagnostics (Section B) for InferenceEngine.Core 1.1.0-dev.13` | 9 modified + 5 new (below) |

### Commit 3 file list (InferenceApp)
Modified: `InferenceApp.UI/App.axaml.cs`, `InferenceApp.UI/InferenceApp.UI.csproj`,
`InferenceApp.UI/MainWindow.axaml.cs`, `InferenceApp.UI/Program.cs`,
`InferenceApp.UI/SettingsWindow.axaml`, `InferenceApp.UI/ViewModels/MainViewModel.cs`,
`InferenceApp.UI/ViewModels/SettingsViewModel.cs`, `InferenceApp.UI/ViewModels/VideoViewModel.cs`,
`Tests/Tests.csproj`.
New: `InferenceApp.UI/Converters/CheckResultConverters.cs`,
`InferenceApp.UI/Diagnostics/DiagInferenceCli.cs`,
`InferenceApp.UI/Services/EngineOptionsFactory.cs`,
`InferenceApp.UI/ViewModels/EngineDiagnosticsViewModel.cs`, `Tests/EngineDiagnosticsTests.cs`.

## Phase 5 — Execute (per repo, in order)

### Lib repo (CWD)
```bash
# Commit 1 — doc
git add -- docs/linux-rocm-config.md
git commit -m "docs: rewrite linux-rocm-config.md for the MIGraphX execution provider"
# Commit 2 — memory bank
git add -- .kilocode/rules/memory-bank/context.md
git commit -m "docs: refresh memory bank context with current state"
```

### InferenceApp repo
```bash
cd /home/nam20485/src/github/nam20485/InferenceApp
dotnet test Tests/Tests.csproj -c Debug   # validation gate; expect 57/57
git add -- InferenceApp.UI/App.axaml.cs InferenceApp.UI/InferenceApp.UI.csproj \
  InferenceApp.UI/MainWindow.axaml.cs InferenceApp.UI/Program.cs \
  InferenceApp.UI/SettingsWindow.axaml InferenceApp.UI/ViewModels/MainViewModel.cs \
  InferenceApp.UI/ViewModels/SettingsViewModel.cs InferenceApp.UI/ViewModels/VideoViewModel.cs \
  Tests/Tests.csproj \
  InferenceApp.UI/Converters/CheckResultConverters.cs \
  InferenceApp.UI/Diagnostics/DiagInferenceCli.cs \
  InferenceApp.UI/Services/EngineOptionsFactory.cs \
  InferenceApp.UI/ViewModels/EngineDiagnosticsViewModel.cs \
  Tests/EngineDiagnosticsTests.cs
git commit -m "feat: expose Engine-Info diagnostics (Section B) for InferenceEngine.Core 1.1.0-dev.13"
```

## Skipped (agent/plan state — NOT committed)
- lib: `.kilo/plans/1783136153190-pr15-merge-to-development.md` (this plan file).
- InferenceApp: `.kilo/plans/1783148927924-load-amd-migraphx-ep.md`, `docs/plans/load-amd-migraphx-ep.md`.

## Phase 6 — Report
- Scan: lib clean; InferenceApp 3 false positives (authorization flagged).
- Validation: lib doc-only (build unaffected); InferenceApp `dotnet test` 57/57.
- 3 commits created (2 lib, 1 InferenceApp). No push (user must explicitly request).

## Risks / notes
- **InferenceApp scan exit ≠ 0** — do NOT commit until the user explicitly authorizes proceeding
  past the 3 confirmed false positives (credential-* variable names + a `cancellationToken` regex hit).
- No secrets are printed or staged.
- Do not push without explicit request.
