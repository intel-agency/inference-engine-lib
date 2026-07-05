# MIGraphX / rocBLAS Host Crash — Troubleshooting (`TensileLibrary.dat` "Illegal seek")

> **Symptom host:** a Debian/Ubuntu + AMD GPU box where the MIGraphX runtimes are installed.

## Symptom

The application (or the `InferenceEngine.Tests` host) **hard-crashes** during engine `LoadAsync` /
the first inference, right after MIGraphX initialization messages, with:

```text
rocBLAS error: Cannot read /opt/rocm/lib/../migraphx/lib/../../rocblas/library/TensileLibrary.dat:
Illegal seek for GPU arch : gfx1031
```

…and the process aborts. In the test runner this surfaces as `Test Run Aborted … Test host process
crashed` with **0 failed tests** (the crash is a host teardown/init abort, not a test failure).

> This is **distinct** from a missing runtime (e.g. `HipRuntimeMissing`), which produces a *graceful*
> CPU fallback. A *present but corrupt* rocBLAS Tensile library **hard-crashes** the process — there
> is no managed fallback from this path.

## What it means

- **Good news:** the MIGraphX execution provider is being appended — `/dev/kfd`, HSA, HIP
  (`libamdhip64.so.7`), and `libmigraphx_c.so.3` are all present. MIGraphX got far enough to start
  compiling its graph.
- **The failure:** rocBLAS (the BLAS library MIGraphX uses for matrix multiplies) cannot read its
  **Tensile kernel cache** (`TensileLibrary.dat`) for your GPU's architecture.
  - `gfx1031` = RDNA2 Navi 22 (e.g. **Radeon RX 6700 / 6700 XT / 6750 XT**).
- **"Illegal seek"** means the file is **corrupt, truncated, zero-length, read-protected**, or a
  **rocBLAS/Tensile build that doesn't ship the kernel for that arch** (version/arch mismatch).

## Diagnose (in order)

```bash
# 1. Confirm the GPU arch (gfx1031 = RX 6700 XT)
rocminfo | grep -E "Name:|Marketing Name:"

# 2. Find the actual TensileLibrary.dat path and check it
ls -l /opt/rocm/rocblas/library/TensileLibrary.dat
#  -> missing, 0 bytes, or not world-readable => corrupt/incomplete install

# 3. Check the installed rocBLAS version + that it matches your ROCm
apt list --installed 2>/dev/null | grep -E "rocblas|rocm-core|migraphx"

# 4. Confirm ROCm coherence (no 5.x/6.x mix)
rocm-smi --showproductname
rocminfo | head
```

A 0-byte or missing `TensileLibrary.dat`, or a `rocblas` whose version doesn't match the installed
`rocm-core`, is the root cause.

## Fixes

1. **Reinstall rocBLAS** (the most common fix — regenerates `TensileLibrary.dat`). Use the AMD ROCm
   apt repo so the version matches your ROCm:
   ```bash
   sudo apt update
   sudo apt install --reinstall rocblas
   # verify the file is now non-zero and readable:
   ls -l /opt/rocm/rocblas/library/TensileLibrary.dat
   ```
   If `rocblas` isn't a standalone package on your distro, reinstall the umbrella that ships it
   (e.g. `sudo apt install --reinstall rocblas rocm-libs`).

2. **Fix permissions** (rare — usually a reinstall resolves it, but if the file exists yet isn't
   world-readable):
   ```bash
   sudo chmod -R a+r /opt/rocm/rocblas/library/
   ```

3. **Arch override** (if the installed Tensile build lacks the gfx1031 codepath):
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=10.3.0   # RDNA2 gfx1030 codepath; widely compatible
   ```
   Retry the app in the same shell.

4. **Re-establish ROCm coherence.** If you mixed ROCm versions (e.g. Debian's HIP 5.x `.so.5` plus
   AMD's ROCm 6.x rocBLAS), the Tensile/HIP pairing is invalid. Reinstall a single coherent ROCm
   stack from the AMD repo:
   ```bash
   sudo apt install --reinstall hip-runtime-amd migraphx rocblas
   ```
   Do **not** symlink `libamdhip64.so.5` → `libamdhip64.so.7` — that triggers a separate
   LLVM-collision abort (see [`linux-rocm-config.md`](linux-rocm-config.md)).

## Verify the fix

After reinstalling/fixing, confirm MIGraphX initializes cleanly:

```bash
# InferenceApp diagnostics CLI (loads both engines, prints provider + check trail):
InferenceApp.UI --diag inference
# Expect: Provider : MIGraphX (device 0)  /  Fallback : no   (no rocBLAS error)
```

or run the library's MIGraphX test (it should now PASS instead of crashing the host):

```bash
dotnet test --filter "FullyQualifiedName~MIGraphXWithAllChecksPassed"
```

## Why it crashes the whole process

rocBLAS calls a fatal/abort path during MIGraphX graph compilation (the first matmul kernel lookup),
which happens **inside native ONNX session construction** — before the managed `BaseInferenceEngine`
fallback can catch it. So unlike a *missing* library (which the diagnostics probe reports and the
engine gracefully falls back from), a *broken* Tensile cache aborts the process outright. Fixing the
rocBLAS install (step 1) is the resolution; the engine code needs no change.

## Related
- [`linux-rocm-config.md`](linux-rocm-config.md) — MIGraphX setup, the `.so.5` ABI warning, and the
  provider-selection check trail (`hip.dlopen` / `migraphx.dlopen` / `migraphx.append`).
- `EngineInfo` / `OnSessionInitialized` — the diagnostics that report *why* MIGraphX did or did not
  load (provider trail + granular `InferenceFallbackReason`).
