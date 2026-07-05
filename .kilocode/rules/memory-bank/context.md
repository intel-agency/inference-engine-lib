# Context

## Current work focus

- **Section-B (consumer diagnostics) implemented in InferenceApp** (repo `nam20485/InferenceApp`, branch `dev/performer-id-adaptation`), consuming `InferenceEngine.Core 1.1.0-dev.13`. EngineDiagnosticsViewModel + OnSessionInitialized wiring for both engines + SettingsWindow diagnostics panel + `--diag inference` CLI + consumer test. Builds clean; 57 tests pass; `--diag inference` verified end-to-end (currently `Provider: Cpu`, `Fallback: HipRuntimeMissing` on this box). **Changes are uncommitted.**
- `1.1.0-dev.13` is published to the intel-agency GitHub Packages feed (manual `workflow_dispatch` of the Publish NuGet workflow after setting `VERSION_PREFIX=1.1`).
- `docs/linux-rocm-config.md` rewritten for MIGraphX (Overview, runtime libs, ONNX-Runtime-side probe, a new `--diag inference` verification section + check table, install note with the `.so.5` ABI warning, behavior/confirming-EP/why-no-throw sections updated to `EngineInfo.Provider`). `libonnxruntime_providers_rocm` refs are gone; minor `IsRocmRuntimeUsable` mentions remain in the troubleshooting/appendix.

## Recent changes (lib)

- WS1 done: `VERSION_PREFIX` bumped `1.0 → 1.1`; `1.1.0-dev.13` published.
- WS2.1 done: `docs/linux-rocm-config.md` MIGraphX rewrite.
- WS2.2 pending (USER, sudo): `sudo apt install hip-runtime-amd migraphx` from the AMD ROCm repo on this Debian box to activate MIGraphX.
- WS2.3 pending: after install, validate via `EngineInfo_LinuxX64WithKfd_MIGraphXWithAllChecksPassed` (expect `Provider == MIGraphX`) and InferenceApp `--diag inference`.

## Key gotcha: NUGET_AUTH_TOKEN

- Both repos' `nuget.config` use `%NUGET_AUTH_TOKEN%` for the `github-intel-agacy` source. If unset in the shell, `dotnet restore`/`pack`/`push` hang (and VS Code C# DevKit's `dotnet restore --interactive` hangs for hours). Fix: export `NUGET_AUTH_TOKEN` (the `ghp…` PAT) in every restore shell; persist in `~/.bashrc`.

## Next steps

- USER: install `hip-runtime-amd` + `migraphx`; then validate GPU (`--diag inference` should show `Provider: MIGraphX`).
- Commit the InferenceApp Section-B changes + the lib doc rewrite when ready.
- (Optional) promote `development → main → release` for a stable `1.1.N` package + NuGet.org.
- (Optional) finish the minor `IsRocmRuntimeUsable` cleanup in `linux-rocm-config.md` troubleshooting/appendix.
