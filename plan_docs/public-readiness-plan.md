# Public Readiness Plan — InferenceEngine.Core (`inference-engine-lib`)

| | |
|---|---|
| **Status** | Proposed — awaiting owner go-ahead for Phase 0 |
| **Goal** | Repo public on GitHub, `InferenceEngine.Core` live on NuGet.org, docs + coverage on GitHub Pages, portfolio-quality presentation |
| **Downstream assumption** | `InferenceEngine.ROCm.Runtime.linux-x64` (ORT 1.24.1 couplet) is published publicly on NuGet.org by the `inference-engine-rocm` readiness effort (its PR #5 / `dev/public-readiness` is in flight; publish plumbing — `NUGET_API_KEY` secret, `release` environment with reviewers — already exists there). First public version will be `1.24.1.{run>36}` (next run after the current stable `1.24.1.36`, which lives only on GitHub Packages). |

## Locked decisions (owner feedback, 2026-10-07)

- **License:** MIT (matches downstream's readiness branch, which already switches its csproj from AGPL to MIT). Both siblings end up MIT — coherent.
- **Authoring:** `Authors=intel-agency`; NuGet.org publication happens under the **intel-agent** account (the API key owner — package *owners* on nuget.org come from the pushing key's account, nothing to change in csproj).
- **Versioning:** keep the standard scheme — `VERSION_PREFIX` repo variable + `github.run_number`, branch-suffixed (`dev` / `rc` / stable), identical to downstream's `build-rocm-linux.yml`.
- **Coverage:** ReportGenerator HTML report **+ PR comment** with the summary.
- **API docs:** generated (DocFX) and published to **GitHub Pages, together with the coverage HTML**.
- **Signing:** none (agreed — NuGet.org repository signing + SourceLink + snupkg is the modern OSS posture).

---

## Direct answers to owner questions

### What does `VERSION_PREFIX` get set to?

**`1.2`** (repo variable, can be set during Phase 0 — dev pushes to GitHub Packages immediately become `1.2-dev.{run}`).

Why `1.2` and not staying at `1.1`:

1. GitHub Packages holds private-only stables `1.1.21` and `1.1.29`. The first *public* stable must not sit inside the same `1.1.x` line as versions nuget.org consumers can never fetch — a minor bump makes the public era unambiguous.
2. SemVer-wise the license change (AGPL → MIT) plus first public API commitment is exactly what a minor bump communicates.
3. Downstream's convention is that `VERSION_PREFIX` tracks the *product's* version (downstream's product is the ORT runtime, hence `1.24.1`; this repo's product is its own API — its version is independent of ORT's).

Run-number note: the current `publish.yml` counter is at **137** — keeping the filename would make the first stable something like `1.2.138`. The workflow gets substantially restructured in Phase 0 anyway (see §0.9); **renaming the file (e.g. `publish-nuget.yml`) resets the run counter**, so the first stable lands at a clean `1.2.{low single digit}` after a few dev runs. Recommended.

### Where does the ROCm runtime package come from today?

**GitHub Packages, authenticated.** `nuget.config` declares the `github-intel-agency` source (`https://nuget.pkg.github.com/intel-agency/index.json`); CI restores with `NUGET_AUTH_TOKEN = secrets.GITHUB_TOKEN`; local development requires a `NUGET_AUTH_TOKEN` env var (PAT with `read:packages`). `InferenceEngine.Core.csproj` pins `InferenceEngine.ROCm.Runtime.linux-x64 1.24.1.36`, which exists **only** on that private, auth-required feed — it is not on nuget.org (404, verified). This is why nothing can ship publicly until the downstream package lands on nuget.org: any public `InferenceEngine.Core` would carry an unresolvable dependency.

### When must the repo go public, and is anything exposed?

**Flip order: Phase 0 merged → flip (Phase 1) → first release push (Phase 2).** The repo must be public *before* the first NuGet.org publish so the package's `projectUrl`/`repository` metadata resolves for every visitor the day the package appears. Docs/homepage go live in Phase 0 (staged inside the private repo — Pages builds fine from private repos for org sites, but the public URL matters at flip time).

Exposure scan verdict (full working tree + all 170 commits across all branches + history patterns + blob sizes): **SAFE TO GO PUBLIC — no secrets, no credentials, no private IPs / VPN hostnames anywhere in history.** One should-fix and some polish, all handled in §0.1:

| Finding | Severity | Fix |
|---|---|---|
| `docs/rocm-wsl2-runner-setup.md` names the owner's machine (workstation model, machine-specific runner name) | should-fix | genericize hostname to `rocm-runner-01` (or delete doc) |
| 5.9 MB binary `plan_docs/docs_linux_rocm_build_guide.md.docx` in history | cosmetic | delete file; history rewrite optional (keeps clone size at +5.9 MB forever otherwise) — recommend skipping the rewrite |
| AI-tool scratch (assistant plan/rules/memory files, editor config, custom-instruction files, `*.code-workspace`) and internal-path docs (`docs/Review___*`, `docs/exception-analysis.md`) — leak local home-directory paths and reference the private consumer repo | polish | delete + `.gitignore` |
| Commit author emails become visible on flip (three personal/work addresses across the commit history) | cosmetic | accept (recommended) or `git filter-repo` rewrite + force-push all branches (not recommended — destroys GPG signature history) |

---

## Phase 0 — start now (nothing blocked on downstream)

> **Status (2026-10-07): COMPLETE — merged to `main`.** Progress log:
> - §0.1–§0.6 — PR #25 (`dev/public-readiness` → `development`), 3-OS CI green, independent reviewer pass per workstream (all findings fixed: conflict markers in this doc, THIRD-PARTY-NOTICES packing, `await using` doc samples).
> - §0.7 — coverage HTML + in-place PR summary comment live-verified on PRs #25/#26 (ReportGenerator 5.5.11; beats baseline: 71.0% line / 56.3% branch).
> - §0.8 — DocFX 2.81.0 site + coverage deployed to Pages from `main` (`docs.yml` dispatch run 37651346985, `github-pages` deployment 16:24Z). Public URL answers at flip. One fix along the way: `upload-pages-artifact` inlined to satisfy the repo's full-SHA action-pinning policy (PRs #27/#28).
> - §0.9 — `publish-nuget.yml` live on `development`: dry-run green (GitHub Packages publish, `create-release` correctly skipped off `release`) — run 37646968358.
> - §0.10 — environments/Pages/Discussions done via API by lead; `VERSION_PREFIX=1.2`, CodeQL default setup (post-flip), `NUGET_API_KEY` remain owner-manual (checklist below).
> - History hygiene: the branch was linearized and fully GPG-signed before merge (`development` ruleset requires linear history + signatures).

Everything below is executable immediately. Execution will be delegated to parallel worker subagents where independent (README, CI workflows, governance files, docs site), integrated and verified here.

### 0.1 Exposure remediation

- Genericize `docs/rocm-wsl2-runner-setup.md` (machine-specific runner name → `rocm-runner-01`, drop the workstation-model line) or delete it.
- Delete: AI-tool scratch (assistant plan/rules/memory files, editor config, custom-instruction files, the `.code-workspace`), `docs/Review___ONNX_Runtime_Missing_Native*`, `docs/Review___ONNX_Runtime_Missing_Native_RESOLUTION.md`, `docs/exception-analysis.md`, `plan_docs/docs_linux_rocm_build_guide.md.docx`.
- Add `.gitignore` entries so they don't return (the AI-tool scratch paths above + `*.code-workspace`).
- Keep: `AGENTS.md`, `.github/copilot-instructions.md` (owner's own AI conventions — fine to publish), `plan_docs/*.md` (portfolio value), `archive/*.seda` (verified clean).
- **Verify:** re-run the exposure scan clean; `git status` shows only intended deletions.

### 0.2 License → MIT

- Add root `LICENSE` (MIT, `Copyright (c) 2026 Artificial Intelligence Agency`), mirroring downstream's readiness-branch text verbatim.
- `InferenceEngine.Core.csproj` + `InferenceEngine.Examples.csproj`: `PackageLicenseExpression>MIT`, pack `LICENSE` into package root (`<None Include="../LICENSE" Pack="true" PackagePath="\" />`) — downstream convention.
- README license section + badge swap.
- `PackageRequireLicenseAcceptance`: recommend **false** (MIT grants permission unconditionally; acceptance dialogs are for restrictive EULAs). Note: downstream currently keeps `true` — flag there for alignment, either way is valid.
- **Verify:** pack → nuspec shows `MIT`, LICENSE at package root.

### 0.3 Package metadata & icon

- Generate `InferenceEngine.Core/icon.png` (512×512, geometric engine/chip glyph — PIL/ImageMagick available locally; owner can later drop in a designed logo, same filename).
- csproj: `<PackageIcon>icon.png</PackageIcon>` + `<None Include="icon.png" Pack="true" PackagePath="\" />`; fix the leading space in `PackageLicenseExpression`; trim `Copyright` to match downstream style.
- `InferenceEngine.Examples.csproj`: `<IsPackable>false</IsPackable>` (example code, never published; currently packs into local output on every build).
- `<EnablePackageValidation>true</EnablePackageValidation>` on Core (baseline rule lands after first stable, Phase 2).
- **Verify:** pack → nuspec contains `icon`, icon renders in a local feed view (nuget.org preview validates on push).

### 0.4 Dependency updates (independent of downstream)

| Package | From → To | Why / risk |
|---|---|---|
| `Microsoft.SourceLink.GitHub` | 8.0.0 → **10.0.401** | kills active NU1902 (GHSA-23fw-v26w-5fgq, transitive `Microsoft.Build.Tasks.Git`) — zero code risk |
| `Microsoft.ML.OnnxRuntime.DirectML` | 1.23.0 → **1.24.1** | version-matched to base ORT (both ship `onnxruntime` natives; mismatch risks subtle native conflicts) |
| `Microsoft.Extensions.Logging.Abstractions` | 10.0.3 → **10.0.12** | patch, zero risk |
| `Microsoft.ML.Tokenizers` | 1.0.3 → **2.0.0** | major — gated on the full test suite passing (text-engine tests cover it); fall back to 1.0.x if red |
| `Microsoft.ML.OnnxRuntime` | 1.24.1 → (defer to Phase 2+) | couples the downstream native couplet; 1.24.1 is the couplet the public ROCm runtime will ship first |

- **Verify:** `dotnet build` zero NU19xx; full `dotnet test` green on all three OSes in CI.

### 0.5 README rewrite + ARCHITECTURE.md

Restructure to the downstream's problem/solution shape (its README is the house style), lead with the gap:

- **The Problem:** every .NET ONNX app reimplements the same plumbing — model download/caching, execution-provider selection and fallback, warmup, tensor pre/post-processing, session lifecycle. ONNX Runtime's .NET API is session-centric and imperative; nothing gives you a *testable engine contract*.
- **The Solution:** `IInferenceEngine<TInput,TOutput>` behind one `LoadAsync`/`PredictAsync` contract — hardware auto-detection (DirectML/CoreML/MIGraphX-with-CPU-fallback), registry-driven model acquisition, warmup, full `IDisposable` lifecycle, plus streaming (`IAsyncEnumerable`) autoregressive decode with KV-cache carry-back.
- Badges: CI, NuGet (goes live at Phase 2), MIT, .NET 10, coverage (goes live with Pages), docs link.
- Sections: Quick Start (runnable YOLO demo), Features table, Architecture (diagram + link to new `ARCHITECTURE.md`), Streaming, Text engines, Model providers, Hardware matrix, Requirements + Versioning tables (mirror downstream), Contributing, License.
- Cut/move to `AGENTS.md`/`CONTRIBUTING.md`: branching model, CI internals, self-hosted-runner notes.
- New `ARCHITECTURE.md` at root (component diagram, provider resolution order, native-library strategy incl. the Linux symlink/override targets) — mirrored from downstream convention.
- **Verify:** all code samples compile; every link resolves; badges point at correct workflows.

### 0.6 Governance files (mirror downstream `dev/public-readiness` set)

`SECURITY.md` (private vulnerability reporting), `CONTRIBUTING.md` (GPG-signed commits, no squash/rebase, issue-first for significant work), `CHANGELOG.md` (seeded with the Phase-0/2 releases), `THIRD-PARTY-NOTICES.txt` (MIT/BSD notices for ORT, Tokenizers, SkiaSharp, SourceLink), `.github/ISSUE_TEMPLATE/bug_report.md` + `config.yml`, `.github/PULL_REQUEST_TEMPLATE.md` (checklist incl. GPG-signed commits).
- **Verify:** GitHub community checklist greens (post-flip).

### 0.7 CI — coverage HTML + PR comments

Extend `build-test.yml` (ubuntu job is the coverage authority):

- Keep `--collect:"XPlat Code Coverage"`; add ReportGenerator (pinned 5.5.x): `-reporttypes:Html;MarkdownSummaryGithub`.
- Upload `coverage-report/` HTML as artifact (rename per-PR, retention 30 d).
- PR comments: `peter-evans/find-comment` + `peter-evans/create-or-update-comment` (SHAs pinned, matching repo practice) posting the `MarkdownSummaryGithub` output, updated in place on re-runs. Current baseline to beat: **68.9% line / 54.3% branch** (Core 72.7%).
- **Verify:** open a draft PR, confirm the coverage comment appears and updates on force-push.

### 0.8 CI — API docs + coverage site on GitHub Pages

- New `docs.yml` workflow (push/PR to `main`, `workflow_dispatch`): DocFX (dotnet tool, pinned) builds from Core's existing XML docs (`GenerateDocumentationFile` already on) + a lightweight conceptual landing page; copy the coverage HTML into the site at `coverage/`; deploy with `actions/upload-pages-artifact` + `actions/deploy-pages`.
- Create the `github-pages` environment (branch policy `main`) — mirrors downstream's environment.
- Site: `https://intel-agency.github.io/inference-engine-lib/` ( Pages is already enabled on the downstream repo, none here — enable via `gh api` in this step).
- Salvage `dev/docfx` branch's `dotnet-tools.json` as the starting point; do not merge that branch wholesale.
- **Verify:** Pages URL serves the API index and `coverage/index.html`.

### 0.9 CI — publish workflow restructure (downstream `create-release` pattern)

Restructure `publish.yml` → **`publish-nuget.yml`** (rename resets the run counter — see VERSION_PREFIX answer):

- Keep: 3-OS build/test gate → pack on ubuntu (`-p:Version=$VERSION` from the same branch-suffix script) → GitHub Packages publish (nupkg + snupkg, `--skip-duplicate`) for `development`/`main`/`release`.
- Add `create-release` job (needs pack, `if: github.ref_name == 'release'`, **`release` environment** — created here with the same reviewer protection as downstream: the owner account, `intelagentone`):
  - `SHA256SUMS.txt` over nupkg + snupkg,
  - GitHub Release `v{VERSION}` via `softprops/action-gh-release` (same pinned SHA as downstream) with `generate_release_notes: true`, assets: nupkg, snupkg, SHA256SUMS,
  - `dotnet nuget push` nupkg + snupkg to `https://api.nuget.org/v3/index.json` with `secrets.NUGET_API_KEY`.
- Remove the `PUBLISH_NUGET_ORG` variable gate entirely — downstream convention is environment-based gating, no toggle var. (The `nuget-org` environment is superseded by `release`; delete it.)
- **Verify:** dry-run on `development` (dev version to GitHub Packages only, release job correctly skipped); workflow-lint YAML.

### 0.10 Repo variables prep

- **`VERSION_PREFIX=1.2`** — owner-console step (Settings → Secrets and variables → Actions → Variables). Agent note: the REST write (`PUT /actions/variables/VERSION_PREFIX`) returns 404 for both available tokens (env `GITHUB_TOKEN` and keyring) despite read access — treat variable writes as owner-manual. Dev pushes stay on the old prefix until this is set, which is harmless.
- Enable **CodeQL default setup** (`gh api`, `security-and-analysis`) — downstream's only scanner convention; free for public repos at flip. **Status: agent API attempt 404s (token lacks scope / GHAS required while private) → owner enables post-flip via Settings → Code security → Code scanning → Default setup.**
- `NUGET_API_KEY` secret: **owner-manual** — create a scoped key on nuget.org under the **intel-agent** account (Push only, package scope `InferenceEngine.Core`), add as repo secret. Needed before Phase 2, not before. (Downstream already created its key — Phase 0 ✅ there.)

**Executed by lead agent (2026-10-07, `gh api`):** `release` environment created (required reviewers: the owner account + `intelagentone`; protected-branch policy — mirrors downstream); `github-pages` environment created (custom branch policy: `main` only); superseded `nuget-org` environment deleted; GitHub Pages enabled (`build_type: workflow`); Discussions enabled (target of issue-template contact link).

---

## Execution model (subagents) — mirrors downstream `Public-Readiness-Master-Plan.md` D9

Subagents run the model configured for their profile (worker/scout/reviewer) in Delta settings — **not** automatically the lead session's model. Verified catalog (2026-10-07, re-verified in this session): `glm-5.3[effort=low|high|max]` and `glm-5.3-flash[effort=low|high|max]`, both defaulting to `effort=low` — **always pass an explicit effort**, or spawns silently run at `low`.

| Role (this repo's workstreams) | Model selector |
|---|---|
| Lead session — integration, PRs/merge train, gated publishes | `glm-5.3[effort=high]` |
| Worker — CI workflows (§0.7–0.9), packaging/csproj (§0.3–0.4), dependency bumps | `glm-5.3[effort=high]` |
| Worker — README/ARCHITECTURE (§0.5), governance files (§0.6), hygiene sweep (§0.1), LICENSE (§0.2) | `glm-5.3-flash[effort=high]` |
| Scout — research (icon tooling, DocFX templates, action SHAs to pin) | `glm-5.3-flash[effort=high]` |
| Reviewer — independent verification of each workstream before PR | `glm-5.3[effort=high]` |

Workstreams §0.1–§0.6 touch near-disjoint files → parallel workers on `dev/public-readiness`; §0.7–§0.9 (workflow surgery) sequential after §0.4 lands (CI must test the updated deps); lead owns the merge train and the Phase 1/2 gated steps. Each workstream gets a reviewer pass before its PR.

---

## Phase 1 — flip to public (gate: Phase 0 merged to `main`)

One-time, owner-executed (I prepare the exact commands):

```bash
# 1. Flip visibility (this is the point of no return for public visibility)
gh repo edit intel-agency/inference-engine-lib --visibility public --accept-visibility-change-consequences

# 2. Metadata: description, homepage (docs site), topics
gh repo edit intel-agency/inference-engine-lib \
  --description "Cross-platform .NET AI inference library — ONNX Runtime behind a generic, testable engine contract with streaming decode and automatic GPU fallback" \
  --homepage "https://intel-agency.github.io/inference-engine-lib/" \
  --add-topic dotnet --add-topic csharp --add-topic onnx --add-topic onnxruntime \
  --add-topic machine-learning --add-topic inference --add-topic gpu --add-topic directml \
  --add-topic coreml --add-topic rocm --add-topic migraphx --add-topic llm --add-topic streaming --add-topic nuget

# 3. Enable CodeQL default setup (agent API 404s while private; free once public)
gh api -X PUT /repos/intel-agency/inference-engine-lib/code-scanning/default-setup \
  -f state=enabled -f query_suite=Default -f 'languages[]=csharp'

# 4. Set the version line (dev pushes become 1.2-dev.{run}; first stable lands at 1.2.{low N} after the publish-nuget.yml counter reset)
#    GitHub UI: Settings → Secrets and variables → Actions → Variables → VERSION_PREFIX = 1.2

# 5. Verify: community profile checklist greens, README badges render,
#    https://intel-agency.github.io/inference-engine-lib/ serves publicly,
#    commit-email visibility is accepted (recommended).
```

Nothing else blocks on visibility — the exposure scan is clear with §0.1 applied.

---

## Phase 2 — blocked on downstream NuGet.org publish

**Gate:** `InferenceEngine.ROCm.Runtime.linux-x64` stable ≥ `1.24.1.37` is live on nuget.org (flatcontainer 200).

1. Bump `InferenceEngine.Core.csproj` ROCm `PackageReference` to the exact first public version.
2. Drop the `github-intel-agency` source from `nuget.config` (restore goes fully public/tokenless) — publishing to GitHub Packages stays possible (the workflow adds its authenticated source at push time; local pushes use `dotnet nuget add source` ad hoc).
3. **Tokenless restore validation:** fresh container/CI agent, `dotnet restore && dotnet test` with *no* `NUGET_AUTH_TOKEN` — proves the public dependency chain resolves (the original hard blocker).
4. Push `main` → `release` (per branch flow `development → main → release`); `create-release` job: reviewers approve → GitHub Release `v1.2.x` + nuget.org publish under intel-agent.
5. Post-publish verification: nuget.org package page (icon, README, MIT, license URL, source-repo links, symbol source stepping), `dotnet add package InferenceEngine.Core` from a clean console app on all three OSes, embedded README renders.
6. Follow-ups (post-first-release): `PackageValidationBaselineVersion` pinned to the first stable; ORT `1.24.1 → 1.30.0` couplet upgrade as its own coordinated project with downstream (its `Multi-Version-Branching-Strategy.md` governs); optionally re-enable `FaceDetectionEngineTests` in CI with a cached model to lift Examples coverage (34.5% today).

---

## Owner-manual checklist (cannot be done by the agent)

- [ ] **Set `VERSION_PREFIX=1.2`** — GitHub UI: repo Settings → Secrets and variables → Actions → Variables tab → edit `VERSION_PREFIX` (API write blocked for agent tokens; verified 2026-10-07)
- [x] nuget.org API key → repo secret `NUGET_API_KEY` — owner confirms the secret is defined in this repo (2026-10-07). Key account/scope proven only at first `release` push; `create-release` fail-fasts if the secret is empty
- [x] Confirm `release` environment reviewers (the owner account, `intelagentone`) — created by lead via API 2026-10-07
- [ ] **Enable CodeQL default setup post-flip** — Settings → Code security → Code scanning → Default setup (agent API 404s while repo is private; free once public)
- [ ] Phase 1 visibility flip + metadata (commands provided)
- [ ] Decide: keep `PackageRequireLicenseAcceptance=true` (downstream's current choice) or accept recommendation `false`
- [ ] Decide: accept commit-email visibility (recommended) vs history rewrite (not recommended)
- [ ] Optional: replace generated `icon.png` with a designed logo (same path)

## Deferred / out of scope

- Authenticode/strong-name signing (agreed: skip)
- ORT 1.30.0 upgrade (Phase 2 follow-up, downstream-coupled)
- Self-hosted ROCm runner as required status check (already documented in README T6; unchanged)
- `InferenceEngine.Examples` as its own published package (stays unpackable, referenced from the repo)

## Verification matrix

| Phase | Gate | Evidence |
|---|---|---|
| 0 | All merged to `development`, CI green 3-OS | `gh run list`, coverage PR comment on the merge PR, local pack inspection (nuspec: MIT, icon, LICENSE; zero NU19xx) |
| 0 | Docs + coverage site live | Pages URL serves both |
| 1 | Repo public, metadata set | community checklist green, badges render |
| 2 | Clean-machine restore tokenless | nuget.org page live, `dotnet add package` works on win/mac/linux, GitHub Release `v1.2.x` with SHA256SUMS |

---

## Initiation prompt (fresh-thread kickoff)

Copy-paste block for a new Delta thread with a clean context. Attach the `inference-engine-lib` repo as a Delta worktree to that thread, run it on `glm-5.3[effort=high]`:

```text
You are the lead agent executing an APPROVED plan: take the private repo intel-agency/inference-engine-lib
(inference-engine-lib Delta worktree; local checkout under the owner's home directory)
to public readiness — repo public, InferenceEngine.Core on NuGet.org, DocFX + coverage on GitHub Pages,
portfolio-quality presentation.

SINGLE SOURCE OF TRUTH: plan_docs/public-readiness-plan.md on branch dev/public-readiness (already pushed to
origin). Read it fully before doing anything. Decisions are locked there (MIT, Authors=intel-agency, NuGet.org
under the intel-agent account, VERSION_PREFIX run-number scheme, no code signing). Do not re-litigate them.

EXECUTE:
1. Work on branch dev/public-readiness (branched from origin/development). PRs target development, per repo
   branch flow (feature → development → main → release). Commit messages conventional; GPG-sign if the local
   git config does. Automated review comments on PRs are blocking.
2. Phase 0 (§0.1–§0.10) is unblocked — start now. §0.1–§0.6 are parallel worker subagents; §0.7–§0.9 workflow
   surgery follows §0.4; §0.10 is mostly owner-manual. Follow the Execution model section for subagent model
   selectors — always pass explicit effort, spawns default to low. Independent reviewer subagent verifies each
   workstream before its PR. Update the plan doc's checkboxes as you complete items; commit progress to the
   branch.
3. Phase 1 (visibility flip) and the owner-manual checklist are OWNER steps — prepare exact commands, do not
   execute them yourself.
4. Phase 2 is BLOCKED until InferenceEngine.ROCm.Runtime.linux-x64 is live on nuget.org (check:
   curl -s -o /dev/null -w "%{http_code}" https://api.nuget.org/v3-flatcontainer/inferenceengine.rocm.runtime.linux-x64/index.json → 200).
   The downstream readiness effort (inference-engine-rocm, PR #5 / dev/public-readiness) owns that. When it
   lands: bump the ROCm PackageReference in InferenceEngine.Core.csproj to the first public version, drop the
github-intel-agency source from nuget.config, prove tokenless restore, then walk main → release for the first
   public stable 1.2.x.

KEY FACTS (verified 2026-10-07): exposure scan CLEAN (no secrets in 170 commits; apply §0.1 remediation);
coverage baseline 68.9% line / 54.3% branch; publish workflow counter at 137 (hence the rename to publish-nuget.yml);
VERSION_PREFIX variable write is owner-console-only; repo has zero secrets today; nuget-org environment is
superseded by release environment (delete it). Read AGENTS.md at repo root for repo conventions before editing.

Report progress as concise phase-gated updates. When blocked on an owner decision, ask one specific question.
```
