# OSSF Scorecard `Pinned-Dependencies` — diagnosis and remediation plan

Repo: `intel-agency/inference-engine-lib`. Scorecard runs on `main` only; clearing this
unblocks the `main → release` promote PR (#88).

Status: **plan approved in principle (Option A), but step 2 of the original draft is
invalid for this repo.** A host-RID blocker was found and empirically verified on
2026-10-24; the corrected plan is in §5. Nothing has been changed yet.

---

## 1. What the alert actually is

It is **not a CodeQL finding.** Alert [#33](https://github.com/intel-agency/inference-engine-lib/security/code-scanning/33)
was produced by **OSSF Scorecard v5.1.1**, rule `Pinned-Dependencies`, and posted to PR #88
by `github-advanced-security[bot]`. It is merely *anchored* to
`.github/workflows/codeql.yml:60` because that line (`run: dotnet restore`) is inside the
PR diff. CodeQL's own jobs (`Analyze (csharp)`, `Analyze (actions)`) both passed.

Scorecard's shell analyzer classifies `dotnet restore` as a `nugetCommand` dependency
(`isDotNetCliRestore`), then `isUnpinnedDotNetCliRestore` scans the argument list for a
literal `--locked-mode` token. No token → *"nugetCommand not pinned by hash"*.

**This is 1 of 6 identical findings**, not a one-off:

| Alert | Location |
| --- | --- |
| #31 / #32 | `build-test.yml:43` / `:84` |
| #33 | `codeql.yml:60` ← the reported line |
| #34 | `docs.yml:45` |
| #35 | `publish-nuget.yml:68` |
| #36 | `rocm-gpu-test.yml:66` |

Scorecard run `40078124614` reports `0 out of 6 nugetCommand dependencies pinned` →
`Pinned-Dependencies` **6/10**, overall score **7.7**. A 7th bare restore exists at
`.github/actions/build-and-test/action.yml:57` and is *not* flagged, consistent with
Scorecard scanning only `.github/workflows/`. Fix it anyway for consistency, otherwise the
finding simply migrates there if Scorecard ever widens its glob.

## 2. Root cause: the repo's CPM is correct, Scorecard's regex is not

`Directory.Packages.props` enables Central Package Management and pins all 16 direct
versions exactly. Scorecard **did** detect CPM — proven by the CPM-specific remediation
suffix in the run log, which then lists *every* version as unfixed:

```text
Warn: nugetCommand not pinned by hash: .github/workflows/codeql.yml:60: ...
: some of dependency versions are not fixes to specific versions:
  1.24.1, 1.24.1, 0.14.0, 1.24.1.44, 10.0.12, 2.0.0, 10.0.401, 18.0.1,
  2.9.3, 3.1.5, 1.5.61, 10.1.0, 3.119.2, 3.119.2, 3.119.2, 3.119.2
```

The cause is `isValidFixedVersion` in Scorecard's `internal/dotnet/properties/properties.go`:

```go
pattern := `^\[[^\[,]+\]$`   // only NuGet BRACKET syntax counts as "fixed"
```

Plain `1.24.1` fails that regex; only `[1.24.1]` passes. So `countUnfixedVersions` → 16,
and `collectPostProcessNugetCPMDependencies` declines to pin. **Identical in v5.1.1 and
current `main`** (both were diffed). Effectively an upstream false positive for any
plain-semver CPM repo. Related upstream: [#5348](https://github.com/ossf/scorecard/issues/5348),
[#5349](https://github.com/ossf/scorecard/issues/5349) — those describe a *different* CPM bug
(`Directory.Build.props` shadowing `Directory.Packages.props` because
`processDirectoryPropsFile` returns `false` to stop at the first well-formed props file).
That bug does **not** apply at `60a7f11`, where no `Directory.Build.props` exists — but
this plan adds one, so §5 must not depend on the CPM branch at all.

## 3. The part of the finding that is genuinely real

Not everything here is a false positive. At `60a7f11`:

- **Zero `packages.lock.json` committed** (`git ls-files` → empty).
- **`RestoreLockedMode` set nowhere**, and no restore passes `--locked-mode`.
- So the **transitive graph floats at restore time** — a real supply-chain gap, independent
  of the regex quirk. CPM pins direct versions only.
- The `Directory.Packages.props` header comment claims *"This central file is what the
  Scorecard Pinned-Dependencies check credits"* — factually wrong per §2. Correct it.

## 4. BLOCKER — `packages.lock.json` is host-RID specific here

The original draft said: *"Regenerate + commit `packages.lock.json` for every csproj (one
plain `dotnet restore`)."* **That breaks CI.** All four projects do this:

```xml
<RuntimeIdentifier Condition="'$(Configuration)' == 'Debug' OR '$(Configuration)' == 'Release'
  ">$([System.Runtime.InteropServices.RuntimeInformation]::RuntimeIdentifier)</RuntimeIdentifier>
```

so the RID is the **host's**, and each project has RID-conditional `PackageReference`s:

- `InferenceEngine.Core` → `Microsoft.ML.OnnxRuntime.DirectML` (`win*`),
  `InferenceEngine.ROCm.Runtime.linux-x64` (`linux-x64`)
- `InferenceEngine.Examples` → `SkiaSharp.NativeAssets.{Linux,Win32,macOS}` (per RID family)
- `InferenceEngine.Tests` / `.IntegrationTests` → inherit Core's graph

One lock file therefore cannot be valid on all three CI legs
(`build-test.yml` and `publish-nuget.yml` both matrix over
`ubuntu-24.04, windows-2025, macos-15`).

**Verified empirically** in a throwaway worktree at `60a7f11` (since removed):

| Test | Command | Result |
| --- | --- | --- |
| B | plain `restore --locked-mode`, same host that generated the lock | **exit 0** — all 4 restored |
| C | `-p:RuntimeIdentifier=win-x64` vs the linux lock | **exit 1** — `NU1004: A new package reference was found Microsoft.ML.OnnxRuntime.DirectML`; `SkiaSharp.NativeAssets.Win32`; `runtime identifiers have changed … lock file's runtime identifiers linux-x64` |
| D | `-p:RuntimeIdentifier=osx-arm64` vs the linux lock | **exit 1** — `NU1004: … SkiaSharp.NativeAssets.macOS`; Core's ROCm reference reported as changed |
| A | `restore -r win-x64 --locked-mode` | **exit 1** — `NU1004: project's runtime identifiers have changed` |

So committing Linux-generated locks and adding `--locked-mode` would fail the
**windows-2025 and macos-15** legs of both matrices.

### 4b. The mitigation works — per-RID lock paths via `NuGetLockFilePath`

The fix for §4 is to give each RID its own lock file. **Verified working.**

> ⚠️ Correction to an earlier draft of this doc: the property is **`NuGetLockFilePath`**,
> *not* `RestorePackagesLockFilePath`. The latter does not exist; NuGet silently ignores
> unknown properties, which produced three false "it's ignored" results (from
> `Directory.Build.targets`, from `.props`, and even from an explicit `-p:`) and a wrongly
> reported blocker. The authoritative name was confirmed by grepping the installed SDK —
> `NuGet.targets:951,979` pass `<NuGetLockFilePath>$(NuGetLockFilePath)</NuGetLockFilePath>`
> into the restore graph entry. Lesson: read the SDK/docs for harness property names rather
> than deriving a plausible one.

Two details that matter:

- `$(RuntimeIdentifier)` is **empty** when `Directory.Build.props` is evaluated — props are
  imported before the csproj body sets the RID. A path built from `$(RuntimeIdentifier)`
  silently yields `Project..lock.json` (note the double dot). Compute the host RID inline
  instead: `$([System.Runtime.InteropServices.RuntimeInformation]::RuntimeIdentifier)`.
- The property must live in `Directory.Build.props`, not `.targets`.

Re-run in a throwaway worktree at `60a7f11`, results:

| Test | Setup | Result |
| --- | --- | --- |
| E | `NuGetLockFilePath` + inline host RID, plain `restore` | **4 locks at `lockfiles/<Proj>.linux-x64.lock.json`**, zero stray `packages.lock.json` |
| F | `restore --locked-mode`, linux locks present (ubuntu leg) | **exit 0**, no `NU1004` |
| G | `restore --locked-mode`, path points at `win-x64` with **no** committed lock (windows/macos leg) | **exit 0** — generates the missing lock, no `NU1004` |
| C/D | *shared* single lock + foreign RID (the §4 approach) | **exit 1**, `NU1004` |

Test G is the load-bearing one: it proves a platform whose lock is not yet committed
**degrades to "unenforced", never "broken build"**. That is what makes the phased rollout
in §5 safe — the windows/macos legs cannot regress while their locks are outstanding.

Root of the §4 conflict, for the record: all four csprojs set

```xml
<RuntimeIdentifier Condition="'$(Configuration)' == 'Debug' OR '$(Configuration)' == 'Release'
  ">$([System.Runtime.InteropServices.RuntimeInformation]::RuntimeIdentifier)</RuntimeIdentifier>
```

so the restore graph is host-specific *by design*. Per-RID lock paths work with that design
instead of against it, which is why no csproj or DirectML/ROCm gating change is needed.

Supporting facts established alongside:

- `--locked-mode` with **no** lock file present → **exit 0** and the lock is created. So a
  missing per-RID lock degrades to "unenforced", never "broken build". This is what makes
  the incremental plan in §5 safe.
- Real drift (hand-added `PackageReference` vs an existing lock) → `NU1004 … can't be run
  in locked mode`, **exit 1**. The enforcement is genuine once a lock exists.
- `dotnet add package` rewrites the lock itself, so `NU1004` fires on hand edits and on
  version bumps whose lock was not regenerated — not on normal Dependabot PRs, which
  update the lock in the same commit.
- Lock files record package id/version/**contentHash**, not source URLs, so a lock
  generated locally is portable to CI despite this host defining 3 sources. The local
  `NU1507` warning (nuget.org, `github-intel-agency`, `forgejo`) comes from the
  machine-level NuGet config, **not** the repo's `nuget.config` (nuget.org only) — do not
  "fix" it with package source mapping on the strength of a local-only condition.

## 5. Corrected plan

Use **per-RID lock file paths** so each platform enforces against its own lock, and land
enforcement incrementally rather than guessing foreign-RID graphs from Linux.

**Phase 1 — infrastructure + linux-x64 enforcement (this PR).**

1. Add `Directory.Build.props` (verified verbatim in a throwaway worktree at `60a7f11`):
   ```xml
   <Project>
     <PropertyGroup>
       <RestorePackagesWithLockFile>true</RestorePackagesWithLockFile>
       <!-- Per-RID locks: every csproj sets RuntimeIdentifier to the HOST rid, and the
            package graph is RID-conditional (DirectML / ROCm / SkiaSharp natives), so one
            shared lock cannot be valid on all three CI platforms (§4). The RID is computed
            inline because $(RuntimeIdentifier) is still empty when Directory.Build.props
            is evaluated — using it here yields "Project..lock.json" (§4b). -->
       <NuGetLockFilePath>$(MSBuildThisFileDirectory)lockfiles/$(MSBuildProjectName).$([System.Runtime.InteropServices.RuntimeInformation]::RuntimeIdentifier).lock.json</NuGetLockFilePath>
     </PropertyGroup>
   </Project>
   ```
2. Generate and commit `lockfiles/*.linux-x64.lock.json` for all 4 projects (one plain
   `dotnet restore` on this host — faithful for `linux-x64` only).
3. Switch **all 7** restore sites to `dotnet restore --locked-mode`
   (`codeql.yml`, `build-test.yml` ×2, `publish-nuget.yml`, `rocm-gpu-test.yml`,
   `docs.yml`, `.github/actions/build-and-test/action.yml`).
4. Correct the `Directory.Packages.props` header comment (§3) to describe the lock files as
   the transitive pin, and note the regen requirement for intentional version bumps.

   Effect: `linux-x64` legs (ubuntu-24.04, self-hosted ROCm runner, CodeQL, docs) become
   genuinely enforced. `windows-2025` / `macos-15` compute a `win-x64` / `osx-arm64` path,
   find no committed lock, generate one and exit 0 (test G) — **unenforced but not broken**.
   Scorecard sees the token on every flagged line and pins all 6.

**Phase 2 — complete cross-platform enforcement (follow-up PR).** Add a temporary job that
runs an unlocked restore on `windows-2025` and `macos-15`, uploads the generated
`win-x64` / `osx-arm64` locks as artifacts, then commit those 8 files. Do **not** synthesise
foreign-RID locks on Linux: `InferenceEngine.Core.csproj:144` and
`InferenceEngine.Tests.csproj:90` gate on `'$(OS)' == 'Unix'`, which `-p:RuntimeIdentifier`
cannot emulate, so a locally-forged Windows lock would not match what the runner computes.

**Complementary, not a substitute — bracket every CPM version** (`Version="[1.24.1]"`).
Verified: all 16 bracketed → `dotnet restore` exit 0 and `dotnet build -c Release`
**0 errors** (7 pre-existing warnings: 5× `NU1507`, 2× `xUnit2013`). This satisfies
`isValidFixedVersion` and closes §2's upstream false positive at its source. It is *real*
but narrower hardening than the locks: `[1.24.1]` is an exact pin, whereas today CPM's bare
`1.24.1` resolves to the floating range `[1.24.1, )` (confirmed in the `NU1004` output:
`"requested": "[1.24.1.44, )"`), so brackets stop silent upward drift of **direct** deps.
They do **not** pin the transitive graph — only the lock files do. Caveats: 16 lines touched,
Dependabot bracket preservation unverified, and `NU1507` is pre-existing (multiple sources +
CPM), not bracket-induced. Recommend as a **separate follow-up PR**, not folded into this one.

**Explicitly not doing:** adding `--locked-mode` alone without any lock file. It would clear
all 6 alerts at zero risk, but it would be scanner-gaming — no transitive pinning would
actually exist. That would misrepresent the security posture.

## 6. Constraints (owner-directed, unchanged)

- Work via PR targeting **`main`**.
- Every commit GPG-signed: `commit.gpgsign=true`, key `B4797B08032A9C92`; verify `%G?` = `G`.
- Code-owner review required for `.github/` paths (nam20485 authors → intelagentone approves).
- Do **not** weaken any pinning, permissions, or gate in existing workflows while editing them.
  These edits touch only the `run:` line of restore steps — no `permissions:`, no `if:`, no
  SHA pin, no status-check name changes.

## 7. Validation and acceptance

Local, before pushing:

- `dotnet restore` (unlocked) generates the 4 `linux-x64` locks.
- `dotnet restore --locked-mode` then succeeds — **exit 0**.
- Foreign-RID simulation must be expected to fail and is **not** a regression:
  `-p:RuntimeIdentifier=win-x64 --locked-mode` → `NU1004` until Phase 2 lands.
- Full gate: `./scripts/agent-self-check.sh` (or `dotnet build -c Release` +
  `dotnet test -c Release --settings coverlet.runsettings`). Coverage baseline must not drop.

CI: all three `build-test.yml` legs and all three `publish-nuget.yml` legs green; CodeQL,
docs, gitleaks, signature gate green.

Acceptance: after merge to `main`, dispatch `scorecard.yml` via `workflow_dispatch`
(alerts only refresh when the default branch is re-analyzed — the same reason `codeql.yml`
exists as Advanced setup). Confirm `Pinned-Dependencies` reports
`6 out of 6 nugetCommand dependencies pinned`, alerts #31–#36 close, and the score rises
from 6/10.
