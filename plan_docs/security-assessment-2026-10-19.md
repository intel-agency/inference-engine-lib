# Security Assessment — inference-engine-lib + NuGet Supply Chain

**Date:** 2026-10-19
**Reviewer:** Independent third-party security review (cross-family model, adversarial posture)
**Scope:** GitHub repo `intel-agency/inference-engine-lib` (workflows, rulesets, environments, collaborators, git history); NuGet packages `InferenceEngine.Core 1.2.12` and `InferenceEngine.ROCm.Runtime.linux-x64 1.24.1.44`; downstream build repo `intel-agency/inference-engine-rocm`; core library code (model download, registry, worker IPC, P/Invoke).
**Method:** Read-only. No modifications to any target. NuGet packages downloaded and diffed against GitHub release assets; GitHub API used for rulesets/environments/collaborators; OSV queried for dependency vulnerabilities.

---

## Verdict

**3 items must be fixed first** — HIGH-1 (protect the `release` branch + self-review gate), HIGH-2 (tag immutability ruleset), MEDIUM-1 (model download integrity + path validation). The first two are ruleset edits; everything else can land as fast-follows.

## Findings Summary

| ID | Severity | Area | Title |
|---|---|---|---|
| HIGH-1 | HIGH | Branch protection | `release` branch is completely unprotected, and it publishes to nuget.org |
| HIGH-2 | HIGH | Tag protection | Release tags are mutable and are in fact re-pointed |
| MEDIUM-1 | MEDIUM | Code | DownloadingModelProvider: no integrity check, path-traversal-capable cache key, partial-download poisoning |
| MEDIUM-2 | MEDIUM | Branch protection | Merge gate is cosmetic: 0 required approvals, no required status checks |
| MEDIUM-3 | MEDIUM | Secrets | Long-lived `NUGET_API_KEY` retained as fallback (lib) / was primary (rocm) |
| MEDIUM-4 | MEDIUM | Build pipeline | ROCm native build trusts mutable upstream references |
| L1 | LOW | Workflows | `${{ github.ref_name }}` interpolated into `run:` blocks |
| L2 | LOW | Workflows | `--store-password-in-clear-text` writes GITHUB_TOKEN to workspace config |
| L3 | LOW | Workflows | `trusted-publish-verify.yml` mints nuget.org keys outside the environment gate |
| L4 | LOW | Commit signing | Release merge commits signed with a key not on any member's GitHub profile |
| I1 | INFO | Feed/trust | No author signature, no ID-prefix reservation — `InferenceEngine.*` namespace open |
| I2 | INFO | Package | ROCm runtime dependency emitted unconditionally for all RIDs |
| I3 | INFO | Package | buildTransitive targets execute on every consumer build (reviewed: benign) |
| I4 | INFO | Docs | SECURITY.md lacks response-time commitment; omits ROCm package |
| I5 | INFO | Org | Lookalike accounts `intelagentone` vs `intelagent1ne` |
| I6 | INFO | Code | Worker IPC is sound (stdin/stdout, 16 MiB frame cap) |
| I7 | INFO | Hygiene | `archive/*.seda` are executable Python packer scripts in a public repo |
| I8 | INFO | Runners | Self-hosted GPU runner guidance for when it is enabled |

---

## Verified clean (so the findings carry weight)

- **Package provenance:** nuget.org `InferenceEngine.Core.1.2.12.nupkg` is byte-identical to the GitHub release asset except nuget.org's added repository signature (expected — repo-signing rewrites the zip). Release asset matches its `SHA256SUMS.txt`. Same for the ROCm package; all three `.so` files match the SHA-256 digests on the `inference-engine-rocm` v1.24.1.44 release. Nuspec `commit` attribute points at the release-branch merge `fb506f9`. `dotnet nuget verify` passes on both.
- **SHA pinning is real:** `sha_pinning_required: true` at repo level; all 13 unique action SHAs across all workflow/action files resolve to their canonical repos (none are fork SHAs). `NuGet/login@v1` and `softprops/action-gh-release@v3.0.3` pins match the annotated tag targets exactly.
- **No `pull_request_target` anywhere; no untrusted `${{ github.event.* }}` in `run:` blocks** (only PR numbers into `with:`). Default `GITHUB_TOKEN` permission is `read`; write scopes are per-job and minimal.
- **Secrets:** secret scanning + push protection enabled on both repos; org requires 2FA. Independent adversarial re-scan of full history (`ghp_`/`github_pat_`/`nupkg_`/PEM blocks, hex tokens >= 32 chars, password/token assignments) found only action SHA pins and variable *names*.
- **Dependencies clean per OSV:** OnnxRuntime 1.24.1, OnnxRuntime.Extensions 0.14.0, Tokenizers 2.0.0, Logging.Abstractions 10.0.12, SourceLink.GitHub 10.0.401.
- **Self-hosted GPU workflow** has no `pull_request` trigger and is gated on an opt-in repo variable (`ROCM_RUNNER_ENABLED`).
- **OIDC trusted publishing** (nuget.org) adopted in `publish-nuget.yml` with the policy bound to owner/repo/workflow/environment.

---

## HIGH-1 — `release` branch is completely unprotected, and it publishes to nuget.org

**Evidence:**
- `GET /repos/intel-agency/inference-engine-lib/rulesets/12671699` ("protected branches", active) includes `refs/heads/main`, `refs/heads/development`, `refs/heads/production` — **`refs/heads/release` is absent**; `refs/heads/production` does not exist as a branch (dead entry).
- No classic branch protection on any branch (`GET .../branches/{main,development,release}/protection` → 404).
- The `release` branch is what `.github/workflows/publish-nuget.yml:143` gates the `create-release` job on (`if: github.ref_name == 'release'`) — the job that pushes to nuget.org and creates the GitHub Release (lines 140-240).
- The `release` environment has required reviewers `nam20485` + `intelagentone` but **`prevent_self_review: false`** and `can_admins_bypass: true` (`GET .../environments`).
- Collaborators with push access: `nam20485` (admin), `odbdesign` (admin), `turing-alpha` (admin), `intelagentone` (admin), `turing-gamma` (**write**).

**Exploitability:** Any of those accounts can push arbitrary commits directly to `release`, delete it, or force-push unsigned history — bypassing the PR requirement, linear history, and signature rules that `main`/`development` have. The only remaining gate is the environment reviewer approval, and because `prevent_self_review` is false and both reviewers are also admins who can push, one compromised account = malicious package on nuget.org. (Contrast: `inference-engine-rocm`'s ruleset 14790432 correctly covers `release`, plus status checks and code scanning.)

**Fix:**
1. Add `refs/heads/release` to ruleset 12671699; remove the dead `refs/heads/production` entry.
2. Set `prevent_self_review: true` on the `release` environment; require 2 reviewers not including the pusher.

## HIGH-2 — Release tags are mutable and are in fact re-pointed

**Evidence:**
- No tag-target ruleset exists (the only ruleset targets branches).
- The publish workflow itself deletes and recreates tags: `gh release delete "v$VERSION" --cleanup-tag --yes || true` (`.github/workflows/publish-nuget.yml:214-217`).
- `v1.2.10` and `v1.2.12` both currently resolve to the *same* commit `fb506f9` — proof tags move.

**Exploitability:** Immutable releases protect the *assets*, but any write-access account can re-point `v1.2.12` at a different commit. Consumers, auditors, and SourceLink all resolve tag -> source; a moved tag silently breaks source provenance for an already-published package.

**Fix:** Add a tag ruleset for `refs/tags/v*` (deny deletion and update). Cut new tags for new versions instead of recycling.

## MEDIUM-1 — DownloadingModelProvider: no integrity check, path-traversal cache key, partial-download poisoning

**Evidence** (`InferenceEngine.Core/ModelProviders/DownloadingModelProvider.cs`):
- Line 51: `Path.Combine(_cacheDirectory, _modelName)` with zero validation — `../../x` or an absolute `modelName` escapes the cache (arbitrary file write with downloaded content). `Path.Combine` with an absolute second argument returns the absolute path.
- Lines 86-93: no SHA-256/checksum verification, no size cap, `FileMode.Create` leaves a partial file on failure; line 54 then serves that file as a valid cache entry forever (`File.Exists` is the only cache check).
- HttpClient (line 21) follows redirects to any host; no scheme/host allowlisting.
- `ModelRegistry` (`ModelRegistry.cs:82`) deserializes `models.json` — which ships as `contentFiles/any/any/models.json` with `copyToOutput=true` into **every consumer's** output directory — and its `Name` flows directly into `DownloadingModelProvider` as the cache filename via `BaseInferenceEngine.GetModelProvider()` (`BaseInferenceEngine.cs:144-149`).

**Exploitability:** A consumer app that lets users add entries to `models.json` (a documented pattern — `CreateDefaultRegistry`, custom registry paths) gives an attacker: arbitrary-URL download (SSRF in server-side consumers) + arbitrary write path + permanent cache poisoning of a file then loaded by a native ONNX parser (memory-unsafe attack surface). Even without traversal, a compromised mirror permanently poisons the per-user cache — nothing is ever re-validated.

**Fix:**
1. Reject `modelName` containing directory separators, rooted paths, or `..`.
2. Add a `sha256` field per registry entry; verify after download, fail closed.
3. Download to a temp file and atomically rename; delete partials on failure.
4. Cap response size; consider pinning redirects to the origin host.

## MEDIUM-2 — Merge gate is cosmetic: 0 required approvals, no required status checks

**Evidence:** Ruleset 12671699 `pull_request` rule has `required_approving_review_count: 0`; there is no `required_status_checks` rule at all. Copilot auto-review (`review_on_push: true`) is not an approval. PRs to `main`/`development` are mergeable with failing CI and zero human review.

**Fix:** Require >= 1 approving review, add the build/test jobs as required status checks, enable "require branches up to date".

## MEDIUM-3 — Long-lived `NUGET_API_KEY` retained as fallback (lib) / was primary (rocm)

**Evidence:** `.github/workflows/publish-nuget.yml:165-179` falls back to the classic `NUGET_API_KEY` secret whenever the OIDC exchange fails (`continue-on-error: true` on the login step). The published ROCm 1.24.1.44 was pushed with the classic key (`build-rocm-linux.yml` at `eebffc0`). A classic nuget.org key is a bearer credential usable from anywhere, valid until expiry, scoped to whatever the `intel-agent` account owns.

**Fix:** Drop the fallback (the OIDC policy is already correctly bound to workflow + environment). Rotate the classic key now. If a classic key must exist, set the shortest available expiry and scope it to exactly the two package IDs.

## MEDIUM-4 — ROCm native build trusts mutable upstream references

**Evidence:** `scripts/compile_onnx_rocm_docker.sh` (at released commit `eebffc0`):
- `git clone -b v1.24.1 https://github.com/microsoft/onnxruntime.git` — mutable tag, no commit pin, no checksum (eigen, by contrast, is commit-pinned).
- Build runs inside `rocm/dev-ubuntu-22.04:7.2.1` — mutable Docker tag, no digest pin.
- The output `.so` overwrites `libonnxruntime.so` for every downstream consumer via the buildTransitive targets in both packages.

**Fix:** Pin the Docker image by digest and ORT by commit SHA. Add `actions/attest-build-provenance` for the `.so` files so NuGet contents can be tied to a specific workflow run.

## LOW

- **L1 — ref_name in run blocks:** `case "${{ github.ref_name }}" in` in `publish-nuget.yml:72` and `build-rocm-linux.yml:85`. Git ref names permit `"` and `$()` — a script-injection sink reachable only with push access. Pass through `env:` instead.
- **L2 — clear-text token storage:** `dotnet nuget add source --store-password-in-clear-text` writes `GITHUB_TOKEN` into the workspace NuGet.config (`publish-nuget.yml:121-126`). Ephemeral runner + job-end revocation bounds it; still unnecessary.
- **L3 — trusted-publish-verify.yml:** `workflow_dispatch` + `id-token: write` lets any write collaborator mint a 1-hour nuget.org key outside the `release` environment gate. If the nuget.org policy is bound to `publish-nuget.yml` only, the exchange should fail — meaning this workflow is testing nothing; if it succeeds, the policy is broader than intended. Verify which.
- **L4 — unverifiable release signatures:** Merge commits on `release` carry GPG key `B5690EEEBB952194`, not among the keys currently published on `nam20485`'s or any member's GitHub profile — those merges render Unverified despite the `required_signatures` rule. Don't remove old signing keys from profiles while history depends on them; re-register or re-sign going forward.

## INFO

- **I1 — Feed/trust:** Packages carry repository signatures only (no author signature); nuget.org shows `verified: false`; no ID-prefix reservation. The `InferenceEngine.*` namespace is open — anyone can register e.g. `InferenceEngine.Runtime.win-x64` and ride the naming. Request prefix reservation (`InferenceEngine.*`) now, while download counts are ~0. Current squat check: no conflicting IDs found.
- **I2 — Package:** The nuspec dependency on `InferenceEngine.ROCm.Runtime.linux-x64` is emitted unconditionally for `net10.0` (the csproj RID condition doesn't survive packing) — Windows/macOS consumers restore the Linux native package too.
- **I3 — buildTransitive targets:** Execute on every consumer build (by design). Reviewed: file copies within NuGet cache/OutDir only, no network, no eval. The `ForceRocmOnnxRuntimeLibOnLinux` glob fallback (`inferenceengine.rocm.runtime.linux-x64/*/...`) accepts any version directory present in the cache — fine while the namespace is owned; another reason for I1.
- **I4 — SECURITY.md:** Decent (private reporting, supported-versions table) but lacks a response-time commitment and omits the ROCm package.
- **I5 — Org:** Confusingly similar accounts `intelagentone` (admin + release reviewer) vs `intelagent1ne` (read-only). Social-engineering/confusion hazard; consider renaming.
- **I6 — Worker IPC:** `InferenceWorkerHost`/`WorkerProtocol` use stdin/stdout child-process framing with a 16 MiB frame cap — same-UID trust boundary, sound design. `WorkerEngineOptions.ModelPath` flows host -> worker unvalidated, but both sides are the same principal.
- **I7 — Hygiene:** `archive/*.seda` files are executable Python packer scripts embedding old source snapshots — no secrets found inside, but odd executable artifacts in a public repo; consider removing.
- **I8 — Self-hosted GPU runner:** When `ROCM_RUNNER_ENABLED` is set, the workflow runs merged code on push (not PR code — correct). When registering the runner: scope it to this repo only (not org), prefer ephemeral runners; it executes code with a packages-read token and GPU access.

---

## Evidence appendix (API snapshots)

- Ruleset 12671699 include list: `refs/heads/main`, `refs/heads/development`, `refs/heads/production`; rules: deletion, non_fast_forward, required_linear_history, required_signatures, pull_request (0 approvals), copilot_code_review. Bypass: RepositoryRole admin, always.
- Environments: `release` = required_reviewers [nam20485, intelagentone], prevent_self_review=false, protected_branches deployment policy; `github-pages` = custom branch policy `main` only.
- Actions permissions: `allowed_actions: all`, `sha_pinning_required: true`; default workflow permissions: `read`.
- Publish run for 1.2.12: `publish-nuget.yml` run 39337031988, `workflow_dispatch` on `release` @ `fb506f9`, success.
- nuget.org owners for both packages: `intel-agent`. No ID prefix reservation.
- OSV: no vulnerabilities for all five direct .NET dependencies at shipped versions.
