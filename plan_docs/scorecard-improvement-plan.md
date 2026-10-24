# OpenSSF Scorecard Improvement Plan — `inference-engine-lib`

| | |
|---|---|
| **Status** | Proposed — awaiting owner go-ahead |
| **Baseline** | **6.9 / 10** — scorecard v5.1.1 @ `d08ed8e`, [public results](https://securityscorecards.dev/viewer/?uri=github.com/intel-agency/inference-engine-lib) (workflow merged in #72, badge added in #74) |
| **Goal** | ≥ 8.5 via high-value fixes and easy wins; skip checks that can't be honestly satisfied for a .NET library |
| **Non-goal** | Gaming checks. Where the fix has independent value (provenance, lockfiles) it's stated; where it doesn't (fuzzing for .NET) we take the 0 and say why. |

## Baseline scorecard (2026-10-22)

| Check | Score | Reason (abridged) | Plan |
|---|---|---|---|
| Signed-Releases | **0** | releases have no signature assets (attestations exist but aren't release assets) | **Tier 2 — cosign release assets** |
| CII-Best-Practices | **0** | no OpenSSF Best Practices badge | **Tier 2 — register + Passing badge** |
| Fuzzing | **0** | no recognized fuzzer | **Skip — no .NET support** |
| Code-Review | **2** | 3/13 recent changesets approved | **Organic** |
| Branch-Protection | **5** | admin bypass, no last-push approval, no code-owner requirement | **Tier 1 — ruleset settings** |
| Security-Policy | **4** | policy exists but "no linked content found" | **Tier 1 — add links** |
| Contributors | **6** | 2 orgs | Skip — not engineerable |
| Pinned-Dependencies | **6** | 0/6 `dotnet` CLI pins (no CPM / lockfile) | **Tier 2 — CPM + lockfiles** |
| SAST | **7** | CodeQL on 11/30 commits | **Tier 1 — broaden PR trigger** |
| Token-Permissions | **9** | top-level `security-events: write` in codeql.yml | **Tier 1 — move to job level** |
| All others (Binary-Artifacts, Maintained, Dependency-Update-Tool, Dangerous-Workflow, License, Vulnerabilities, Packaging, CI-Tests) | 9–10 | — | Keep |

---

## Tier 1 — easy + high value (no-code or trivial PRs)

### 1. Branch protection settings → 5→~9 · owner action, ~15 min

On the `main` ruleset (mirror on `development` and `release` where it exists):
- **Do not allow bypassing the above settings** (apply to administrators)
- **Require review from Code Owners**
- **Require approval of the most recent reviewable push**
- Keep required checks: Signature Gate, CodeQL, Build & Test

**Prereq (PR):** add `.github/CODEOWNERS` — the code-owner toggle has nothing to resolve without it. Start with `* @nam20485` plus a CODEOWNERS entry for `.github/workflows/` if a second reviewer exists.

### 2. SECURITY.md links → 4→10 · trivial PR

Scorecard's "no linked content found" fires because the disclosure contact is prose, not a link. Make "Report a vulnerability" a real link to `https://github.com/intel-agency/inference-engine-lib/security/advisories/new` and add anchored section links (supported versions, timelines).

### 3. Token-Permissions → 9→10 · trivial PR

`codeql.yml`: delete the top-level `security-events: write` and grant it on the `analyze` job instead (where it's actually consumed). The `publish-nuget.yml` job-level `contents: write` is genuinely required for releases — already correctly scoped, leave it.

### 4. SAST coverage → 7→10 · trivial PR

`codeql.yml`: drop the branch filter on `pull_request:` so **every** PR is scanned regardless of base branch. The 11/30 ratio reflects branches CodeQL never saw (pre-#71 history and feature-branch targets); scanning all PRs plus the weekly schedule converges to full coverage as history accumulates.

---

## Tier 2 — highest absolute value

### 5. Signed-Releases → 0→10 · one workflow change

`actions/attest-build-provenance` already attests the shipped nupkgs (verifiable via `gh attestation verify`), but the Scorecard check only counts **signature files among release assets** (`.sig`, `.sigstore`/`.bundle`, …). In `create-release` (OIDC `id-token: write` already granted):

1. Add a step after checksums: `cosign sign-blob --yes <file>` (keyless) for each `*.nupkg`, `*.snupkg`, `SHA256SUMS.txt`, emitting `<file>.sig` + `<file>.bundle`.
2. Add the `.sig`/`.bundle` files to the release `files:` list.
3. Document verification in the release body template: `cosign verify-blob --bundle <file>.bundle --certificate-identity ... <file>`.

Independent value beyond the score: consumers get offline-verifiable sigstore signatures on release bytes.

### 6. Pinned-Dependencies → 6→10 · medium PR, reproducibility win

Scorecard's own remediation message asks for Central Package Management or a lockfile:
1. Add `Directory.Packages.props` (CPM) and move all `<PackageReference>` versions out of csprojs.
2. Commit `packages.lock.json` per project (`<RestorePackagesWithLockFile>true</RestorePackagesWithLockFile>`).
3. CI restores with `--locked-mode` in `build-test.yml`, `codeql.yml`, `docs.yml`, `publish-nuget.yml`, `rocm-gpu-test.yml` (the six flagged `dotnet` commands).

Dependabot supports CPM, so updates keep flowing.

### 7. CII-Best-Practices → 0→10 · owner action ~1–2 h + badge PR

Register the project at <https://bestpractices.coreinfrastructure.org/> and complete the **Passing** tier checklist. Most answers already true: license, CI tests, security policy, SAST, vulnerability handling, packaging. Then add the badge markdown to the README next to the Scorecard badge.

---

## Tier 3 — organic or skip

- **Code-Review (2):** rises automatically now every change flows through approved PRs under the post-#70 rulesets. Re-check in ~3 weeks; no direct fix.
- **Contributors (6):** a fact about contributor orgs, not something to engineer. Skip.
- **Fuzzing (0):** Scorecard credits recognized engines (OSS-Fuzz, libFuzzer, AFL, Jazzer, …) — none support .NET. A SharpFuzz harness is worthwhile *hardening* but scores nothing; take the 0, revisit if Scorecard adds .NET support.

## Maintenance (not scored)

- **CodeQL Action v3 → v4** before Dec 2026 deprecation — warning already appears in scorecard run logs (`github/codeql-action@…# v3.39.1` in `codeql.yml` + `upload-sarif` in `scorecard.yml`).

## Sequencing

| # | Item | Type | Lands on |
|---|---|---|---|
| A | CODEOWNERS | PR → `main` | enables ruleset toggles in 1 |
| B | SECURITY.md links | PR → `main` | independent |
| C | codeql.yml: job-level perms + all-branch PR trigger | PR → `main` | 3 + 4 |
| D | Ruleset settings (admin bypass, code owners, last-push approval) | owner (repo settings) | after A |
| E | cosign release signatures | PR → `main` | next release verifies |
| F | CPM + lockfiles + `--locked-mode` | PR → `main` | biggest diff; do last |
| G | CII Passing badge registration + README badge | owner + small PR | any time |

**Projected:** 6.9 → **~8.6–9.0** (all Tier 1+2 land: +1.0 Signed-Releases, +1.0 CII, +0.6 Security-Policy, +0.4 Branch-Protection, +0.4 Pinned-Dependencies, +0.3 SAST, +0.1 Token-Permissions, Code-Review drift up organically).
