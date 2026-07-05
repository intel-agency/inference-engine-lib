# Linux ROCm Configuration

## Overview

This guide covers enabling GPU hardware acceleration for ONNX Runtime inference on
Linux x64 hosts that have an AMD GPU installed, using AMD's **MIGraphX** execution
provider. (The older ROCm execution provider is **deprecated and removed as of ONNX
Runtime 1.24.1**; MIGraphX is its successor and is what this library uses.) The
library auto-detects the ROCm/MIGraphX stack at engine load and records a granular,
**ordered diagnostic trail** (`InferenceEngineInfo.Checks`) covering every step of
provider selection — `/dev/kfd`, the HSA userspace runtime, the HIP runtime
(`libamdhip64.so`), the MIGraphX runtime (`libmigraphx_c.so`), provider append, and
session construction. On any host where a required layer is missing, the engine falls
back to CPU and reports a precise `InferenceFallbackReason` (e.g.
`HipRuntimeMissing`, `MIGraphXRuntimeMissing`) instead of failing. The same
auto-fallback shape is used on Windows (DirectML) and macOS (CoreML) — a developer
that ships a cross-platform agent does not need to fork their engine code per OS.

> **Verify in one command.** InferenceApp ships a `--diag inference` subcommand that
> loads both engines and prints the selected provider, the fallback reason, and the
> full check trail. See
> [Verifying GPU Acceleration](#verifying-gpu-acceleration) below. The library exposes
> the same data programmatically via `BaseInferenceEngine.EngineInfo` and the
> `InferenceEngineOptions.OnSessionInitialized` callback.

For the design rationale behind the native-library plumbing, see
[`docs/Review___ONNX_Runtime_Missing_Native_RESOLUTION.md`](./Review___ONNX_Runtime_Missing_Native_RESOLUTION.md).

---

## System Requirements

ROCm support is the conjunction of four independent layers. **All four must be true**
for the library to initialize the ROCm execution provider; if any one is missing the
runtime check logs an info-level message ("ROCm runtime not available on this host —
using CPU provider") and the engine runs CPU-only with no functional regression.

### Hardware

- A discrete AMD GPU from a ROCm-supported family. The current library has been
  verified against RDNA 2 (Navi 21/22/23/24, e.g. RX 6800/6900/6700 XT) and RDNA 3
  (Navi 31/32/33, e.g. RX 7900 XTX); it should also work on CDNA accelerators
  (Instinct MI100/MI200/MI300) and GCN 5th-gen+ parts (Vega 10/12/20).
- Many older and low-end consumer GPUs are not on the ROCm support matrix (RDNA 1
  Navi 10/14, Polaris, pre-Vega GCN, APUs without dGPUs). An RDNA 1 GPU can sometimes
  be coerced into working with `HSA_OVERRIDE_GFX_VERSION=10.3.0` (see below) but
  performance is best-effort and not a supported configuration.
- The canonical, current AMD-published support list lives at
  <https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html>. Cross-reference
  this with the kernel/ROCm version you plan to install before buying new hardware.

### Kernel & Drivers

- Linux kernel 5.x or newer (the `amdgpu` driver is in mainline since 5.0; the
  `amdgpu` module is automatically built and shipped by every major distro's kernel
  package). Kernels 6.x are fine.
- The `amdgpu` driver must be loaded. The simplest check is `lsmod | grep amdgpu`;
  if the module is blacklisted or absent, ROCm cannot run.
- A working KFD (Kernel Fusion Driver) node at `/dev/kfd`. KFD is the in-kernel
  surface that ROCm userspace uses to enqueue work onto the GPU. If `/dev/kfd` is
  missing, the engine will fall back to CPU even if `/dev/dri/renderD128` is
  present.
- A DRI render node at `/dev/dri/renderD128` (and possibly `/dev/dri/renderD129` for
  a second GPU). The render node is the userspace-facing surface for buffer
  submission.
- The current user must have read/write access to `/dev/kfd` and the DRI render
  nodes. This is normally achieved by membership in the `video` and `render` groups;
  many distros create these groups automatically when the `amdgpu` driver packages
  are installed, but they do not always add the current user. Running as root also
  works but is not recommended. **Wrong permissions are the single most common
  cause of "ROCm is installed but the library does not use it."**

### ROCm Userspace Packages

The MIGraphX execution provider needs a narrow set of userspace libraries. The
library's provider-selection probe loads each in order and reports a granular
`InferenceFallbackReason` for the first one that is missing:

- `libhsa-runtime64.so` (the HSA userspace runtime) — provided by `hsa-rocr-dev` /
  `hsa-runtime64` (distro-dependent).
- `libamdhip64.so.7` (**HIP 7.x** — the C/HIP host runtime). This is the single most
  common point of failure on Debian-derived hosts: Debian 13 ships **HIP 5.7**
  (`libamdhip64.so.5`), which is **ABI-incompatible** with the MIGraphX-enabled
  `libonnxruntime.so` the library ships (built against HIP 7.x). Installing Debian's
  `libamdhip64-5` package is **not** sufficient and will report `HipRuntimeMissing`.
  **Do not** symlink `.so.5` → `.so.7` — that produces a hard process abort
  (`LLVM ERROR: Option '…' registered more than once`) because both libs embed
  conflicting LLVM command-line option tables.
- `libmigraphx_c.so.3` (the MIGraphX C API) — provided by the `migraphx` package.

The two runtime packages that satisfy all of the above are:

```bash
sudo apt install hip-runtime-amd migraphx   # from the AMD ROCm apt repo
```

These come from AMD's `amdgpu-install` repository (see
[Installing ROCm](#installing-rocm) below); the distro-packaged equivalents lag the
required version and are only suitable for development, not the ABI MIGraphX needs.

Environment variables that frequently need to be set when running the application:

- `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64` (some ROCm installs only put
  libraries in `/opt/rocm/lib`, some only in `/opt/rocm/lib64`; setting both is
  harmless and covers the union). Modern `.deb`/`.rpm` installs register
  `/etc/ld.so.conf.d/rocm.conf` and run `ldconfig` automatically, so this is usually
  a no-op but is the first thing to check if the probe fails.
- `HSA_OVERRIDE_GFX_VERSION=<gfx-version>` — for older or unsupported GPU families
  that do not have native codepaths in the installed ROCm. Common values:
  - `HSA_OVERRIDE_GFX_VERSION=10.3.0` for RDNA 1 (RX 5000 series, e.g. Navi 10).
  - `HSA_OVERRIDE_GFX_VERSION=9.0.0` for Vega 10 (RX Vega 56/64, MI25).
  Setting this unlocks GPU codepaths at the cost of some operations falling back
  to MIOpen CPU algorithms. See the
  [Troubleshooting](#troubleshooting) section for the diagnostic toggles
  (`MIOPEN_LOG_LEVEL=3`, `HSA_ENABLE_SDMA=0`) that confirm whether a fallback is
  happening.
- `ROCM_PATH=/opt/rocm` (only needed if ROCm is installed somewhere other than the
  default `/opt/rocm`).

### ONNX Runtime Side

The library handles the native-library plumbing for you. There are no manual
edits to `.csproj` files or environment variables required at build time. The
relevant pieces, for context:

- The `InferenceEngine.ROCm.Runtime.linux-x64` NuGet package (currently pinned at
  **`1.24.1.36`**) is referenced on `linux-x64` builds. It ships a single
  **MIGraphX-enabled** `libonnxruntime.so` — unlike the deprecated ROCm EP, MIGraphX
  is compiled into the core ORT shared library, so there is no separate
  `libonnxruntime_providers_*.so` plugin to locate.
- A post-Build MSBuild target (`RestoreUpstreamCpuLibOnLinux`) and a P/Invoke
  symlink target (`SymlinkLinuxNativeLibsForPInvoke`) ensure the right native
  library is resolvable at runtime. (The full rationale is in the
  `Review___…_RESOLUTION.md` doc linked above.)
- Provider selection lives in `BaseInferenceEngine.TryRocmBranch()` (Linux x64). It
  runs an **ordered diagnostic trail**, appending a `ProviderDiagnosticCheck`
  (Pass/Fail/Skip) at each step:
  1. `gpu.user-enabled` — `UseGpuAcceleration` is on.
  2. `rid.platform` — host is Linux x64.
  3. `kfd.present` — `/dev/kfd` exists.
  4. `hsa.dlopen` — `libhsa-runtime64.so` loads.
  5. `hip.dlopen` — `libamdhip64.so.7` loads (the most common failure on Debian).
  6. `migraphx.dlopen` — `libmigraphx_c.so.3` loads.
  7. `migraphx.append` — `AppendExecutionProvider_MIGraphX` succeeds.
  8. `migraphx.session` — the GPU `InferenceSession` constructs.
  The first failure (or a skipped precondition) ends the GPU attempt; the engine
  then builds a CPU-only session and appends `cpu.fallback-options` /
  `cpu.fallback-session` checks. The whole trail is exposed on
  `BaseInferenceEngine.EngineInfo` (provider, device id, `IsFallback`,
  `FallbackReason`, and the ordered `Checks` list) and delivered via the
  `InferenceEngineOptions.OnSessionInitialized` callback. The possible
  `InferenceFallbackReason` values on Linux are `KfdDeviceMissing`,
  `HsaRuntimeUnloadable`, `HipRuntimeMissing`, `MIGraphXRuntimeMissing`, and
  `NativeLibraryLoadFailed` (the last is a catch-all for a recoverable failure
  during GPU session construction).

For the full timeline of the native-library fixes (the original 5 review comments
plus the 4 additional root-cause fixes for upstream package quirks, SkiaSharp
split packages, and the SIGSEGV-on-non-ROCm-hosts hazard), see
[`docs/Review___ONNX_Runtime_Missing_Native_RESOLUTION.md`](./Review___ONNX_Runtime_Missing_Native_RESOLUTION.md).

---

## Verifying Your System

### Verifying GPU Acceleration

The fastest end-to-end check is the diagnostics API. In the consumer app:

```bash
# InferenceApp: loads both engines and prints provider + fallback + check trail
InferenceApp.UI --diag inference
```

On a host that is **not yet GPU-ready** (e.g. a stock Debian box that only ships the
ABI-incompatible `libamdhip64.so.5`), the output pinpoints the exact missing layer
and its install command:

```text
=== Detection (YOLO) ===
  Platform : LinuxX64
  Provider : Cpu
  GPU req  : True
  Fallback : HipRuntimeMissing
  Checks   :
    PASS [0] gpu.user-enabled: UseGpuAcceleration is enabled
    PASS [1] rid.platform: Host is Linux x64
    PASS [2] kfd.present: /dev/kfd exists
    PASS [3] hsa.dlopen: libhsa-runtime64.so is loadable
    FAIL [4] hip.dlopen: libamdhip64.so.7 not found. HIP runtime is required for
            MIGraphX GPU acceleration. Install: sudo apt install hip-runtime-amd
            (from AMD ROCm repository). ...
    SKIP [5] migraphx.dlopen: precondition hip.dlopen failed
    SKIP [6] migraphx.append:    precondition hip.dlopen failed
    SKIP [7] migraphx.session:   precondition hip.dlopen failed
    PASS [8] cpu.fallback-options: CreateCpuOnlySessionOptions() succeeded
    PASS [9] cpu.fallback-session: CPU InferenceSession constructed
OK: at least one engine loaded.
```

Once `hip-runtime-amd` and `migraphx` are installed (see
[Installing ROCm](#installing-rocm)), the same command reports `Provider: MIGraphX`
with all `hip.*` / `migraphx.*` checks at `PASS` and no fallback. The same data is
available programmatically via `engine.EngineInfo` (in any engine deriving from
`BaseInferenceEngine`) and the `InferenceEngineOptions.OnSessionInitialized` callback.

The per-check meaning, for interpreting the trail:

| Check | Meaning |
|---|---|
| `kfd.present` | `/dev/kfd` exists and is accessible. |
| `hsa.dlopen` | `libhsa-runtime64.so` loads. |
| `hip.dlopen` | `libamdhip64.so.7` (HIP 7.x) loads. |
| `migraphx.dlopen` | `libmigraphx_c.so.3` loads. |
| `migraphx.append` | `AppendExecutionProvider_MIGraphX` succeeded. |
| `migraphx.session` | The GPU `InferenceSession` constructed. |
| `cpu.fallback-*` | Appended only on fallback; confirms CPU still works. |

---

### System-level checks

Run the steps below in order on the target host. Each step has a single command, an
expected output hint, and a one-line failure mode description. The consolidated
script at the end runs all seven and reports a single pass/fail summary.

### Step 1 — AMD GPU visible to kernel

The PCI device must be enumerated. This is the most basic check: if the kernel
cannot see the GPU, nothing downstream will work.

```bash
lspci | grep -i amd
```

Expected output: at least one line matching `VGA compatible controller`,
`3D controller`, or `Display controller` with `AMD` or `Advanced Micro Devices`
in the description. Example:

```text
0a:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 21 [Radeon RX 6900 XT]
```

Failure mode: empty output means the GPU is not in a PCIe slot the kernel can see.
Re-seat the card, check BIOS IOMMU settings, or check `dmesg` for PCIe link errors.

### Step 2 — `amdgpu` driver loaded

The driver must be present in the running kernel. Kernels since 5.0 ship
`amdgpu` as a built-in or loadable module; the question is whether it actually
loaded for this device.

```bash
lsmod | grep amdgpu
```

Expected output: a single line beginning with `amdgpu` followed by the module
size and reference count. Example:

```text
amdgpu               9871360  0
```

Failure mode: empty output. The module is either blacklisted (`/etc/modprobe.d/`
often has a `blacklist amdgpu.conf` left over from a previous NVIDIA install) or
the kernel version is too old.

### Step 3 — `/dev/kfd` exists and is accessible

The KFD node is the in-kernel surface ROCm userspace uses. The user must be able
to open it for both read and write.

```bash
ls -al /dev/kfd /dev/dri/*
```

Expected output: `/dev/kfd` is present and the current user (or the `video` and
`render` groups) has `rw` access. Example:

```text
crw-rw---- 1 root video  234,   0 Jun  4 12:00 /dev/kfd
crw-rw---- 1 root video  226, 128 Jun  4 12:00 /dev/dri/card0
crw-rw-rw- 1 root render 226, 128 Jun  4 12:00 /dev/dri/renderD128
```

Failure mode: missing device, or mode/owner is `crw------- root root`. The fix
is to add the user to the `video` and `render` groups:

```bash
sudo usermod -aG video,render $USER
```

then log out and back in.

### Step 4 — DRI render nodes exist

ROCm uses the render node for buffer submission, not the legacy card node. At
least `/dev/dri/renderD128` must exist (a second GPU adds `renderD129`).

```bash
ls -al /dev/dri/renderD128
```

Expected output: a character device file at `/dev/dri/renderD128` with
group `render` (or `video`) and mode `crw-rw-rw-` or `crw-rw----`. Example:

```text
crw-rw-rw- 1 root render 226, 128 Jun  4 12:00 /dev/dri/renderD128
```

Failure mode: file missing entirely. The `amdgpu` driver is loaded but the render
node was not created. This usually means a permissions problem at driver init
(check `dmesg | grep amdgpu` for `drm:amdgpu_device_init` errors) or a kernel
built without `CONFIG_DRM_AMDGPU` set to `=y` or `=m`.

### Step 5 — ROCm HSA runtime library resolvable

The provider-selection probe (`hsa.dlopen` in the diagnostics trail) calls
`NativeLibrary.TryLoad("libhsa-runtime64.so", ...)`. The library must be
discoverable by the dynamic linker.

```bash
ldconfig -p | grep libhsa-runtime64
```

Expected output: one or more lines naming the SONAME and a fully-qualified path.
Example:

```text
libhsa-runtime64.so.1 (libc6,x86-64) => /opt/rocm/lib/libhsa-runtime64.so.1
libhsa-runtime64.so    (libc6,x86-64) => /opt/rocm/lib/libhsa-runtime64.so
```

Failure mode: empty output. Either the `hsa-rocr-dev` package is not installed,
or `/opt/rocm/lib` is not registered with `ldconfig`. Fix with:

```bash
echo '/opt/rocm/lib' | sudo tee /etc/ld.so.conf.d/rocm.conf
sudo ldconfig
```

or, if you cannot modify system config, set `LD_LIBRARY_PATH` in the
application's environment (see [System Requirements](#system-requirements) above).

### Step 6 — ROCm user libraries in library path

Beyond `libhsa-runtime64.so`, the ROCm provider needs `libamdhip64.so`,
`libmiopen.so`, and `librocblas.so` resolvable at session-creation time. Check
all four in one pass.

```bash
ldconfig -p | grep -E 'libhsa-runtime64|libamdhip64|libmiopen|librocblas'
```

Expected output: at least one line per library. Example:

```text
libhsa-runtime64.so.1   (libc6,x86-64) => /opt/rocm/lib/libhsa-runtime64.so.1
libamdhip64.so.5        (libc6,x86-64) => /opt/rocm/lib/libamdhip64.so.5
libmiopen.so.1          (libc6,x86-64) => /opt/rocm/lib/libmiopen.so.1
librocblas.so.0         (libc6,x86-64) => /opt/rocm/lib/librocblas.so.0
```

Failure mode: any line missing. Install the corresponding package
(`miopen-hip`, `rocblas`, `hsa-rocr-dev`, `hip-runtime-amd`) and re-run
`ldconfig`.

### Step 7 — End-to-end ROCm runtime probe

This is the test the library itself runs. Build the project and load any
engine; the log line distinguishes a successful ROCm path from a CPU fallback.

```bash
dotnet build --runtime linux-x64 InferenceEngine.Core/InferenceEngine.Core.csproj
```

Then verify the library's own provider selection via the diagnostics API (or, in
InferenceApp, `--diag inference`). A successful MIGraphX path reports:

```text
Provider : MIGraphX (device 0)
GPU req  : True
Fallback : no
Checks   :
    PASS [0] gpu.user-enabled
    PASS [1] rid.platform
    PASS [2] kfd.present
    PASS [3] hsa.dlopen
    PASS [4] hip.dlopen
    PASS [5] migraphx.dlopen
    PASS [6] migraphx.append
    PASS [7] migraphx.session
```

i.e. `EngineInfo.Provider == InferenceProvider.MIGraphX` with no fallback. A
fallback path (which is **not an error** — it is by design) reports `Provider: Cpu`
with `IsFallback=true` and a `FallbackReason` (e.g. `HipRuntimeMissing`) naming the
first missing layer, plus `PASS` `cpu.fallback-*` checks confirming CPU still works.

If you see a fallback, read the first `FAIL` check in the trail (e.g. `hip.dlopen`)
and re-run the matching system-level step above to fix it. The library decides the
provider once at `LoadAsync` time and will not re-attempt GPU on a later call.

### Consolidated check script

The script below runs all seven steps non-interactively, prints a one-line
result per step, and exits 0 only if every step passes. Paste it into a
file (e.g. `~/check-rocm.sh`), `chmod +x` it, and run it. It is safe to run as
a non-root user; the only side effects are read-only filesystem probes.

```bash
#!/usr/bin/env bash
# check-rocm.sh — verify that an inference-engine-lib host is ROCm-ready.
# Exits 0 on full pass, 1 on any failure. Safe to run as a non-root user.

set -u

pass=0
fail=0
skip=0

step() {
  local name="$1"; shift
  echo
  echo "=== $name ==="
  # Run the inner command and capture its exit code BEFORE the `if` consumes
  # it. We distinguish exit 0 (pass), exit 2 (intentional skip), and any
  # other non-zero (fail).
  "$@"
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "PASS: $name"
    pass=$((pass + 1))
  elif [ "$rc" -eq 2 ]; then
    echo "SKIP: $name"
    skip=$((skip + 1))
  else
    echo "FAIL: $name (exit $rc)"
    fail=$((fail + 1))
  fi
}

# Step 1: AMD GPU visible to kernel
step "Step 1: AMD GPU visible to kernel" bash -c '
  lspci | grep -i amd | grep -Eq "VGA|3D|Display"
'

# Step 2: amdgpu driver loaded
step "Step 2: amdgpu driver loaded" bash -c '
  lsmod | grep -q "^amdgpu "
'

# Step 3: /dev/kfd exists and is accessible
step "Step 3: /dev/kfd exists and is accessible" bash -c '
  [ -e /dev/kfd ] && [ -r /dev/kfd ] && [ -w /dev/kfd ]
'

# Step 4: DRI render nodes exist
step "Step 4: DRI render nodes exist" bash -c '
  [ -e /dev/dri/renderD128 ]
'

# Step 5: ROCm HSA runtime library resolvable
step "Step 5: ROCm HSA runtime library resolvable" bash -c '
  ldconfig -p | grep -q "libhsa-runtime64"
'

# Step 6: ROCm user libraries in library path
step "Step 6: ROCm user libraries in library path" bash -c '
  for lib in libhsa-runtime64 libamdhip64 libmiopen librocblas; do
    ldconfig -p | grep -q "$lib" || { echo "missing: $lib"; exit 1; }
  done
'

# Step 7: End-to-end ROCm runtime probe (the library's auto-detect)
# This step requires the .NET 10 SDK and a built InferenceEngine.Core.
# Comment out this block if you only want to verify the system-level checks.
step "Step 7: End-to-end ROCm runtime probe (requires .NET 10 SDK + ROCm host)" bash -c '
  if ! command -v dotnet >/dev/null 2>&1; then
    echo "dotnet not found on PATH; skipping Step 7"
    exit 2   # 2 = skipped (not a hard failure)
  fi
  if [ ! -d InferenceEngine.Core ]; then
    echo "InferenceEngine.Core not found in CWD; skipping Step 7"
    exit 2
  fi
  dotnet build --runtime linux-x64 -c Release InferenceEngine.Core/InferenceEngine.Core.csproj >/dev/null 2>&1 || {
    echo "dotnet build failed; cannot run runtime probe"
    exit 1
  }
  # The library logs the decision line on LoadAsync; we cannot easily
  # invoke it from bash, so this step is a best-effort build check. To
  # exercise the runtime probe, run the example engine in C# (see
  # docs/linux-rocm-config.md, "Configuring Your Application").
  echo "Build succeeded; runtime probe must be exercised from a C# host."
  exit 0
'

echo
echo "==============================="
echo "ROCm readiness: $pass passed, $skip skipped, $fail failed"
echo "==============================="

[ "$fail" -eq 0 ]
```

A clean run prints seven `PASS` lines (or six `PASS` and one `Skipped` for Step 7
if `dotnet` is not on the PATH) and exits 0. Any `FAIL` indicates a missing
prerequisite; refer to the per-step "Failure mode" notes above to fix it.

---

## Installing ROCm

> **What MIGraphX actually needs:** just `hip-runtime-amd` (provides
> `libamdhip64.so.7`) and `migraphx` (provides `libmigraphx_c.so.3`). The full
> `--usecase=rocm` install below pulls these in (plus much more — MIOpen, rocBLAS,
> etc.) and is the safe choice; for a minimal install run
> `sudo apt install hip-runtime-amd migraphx` once the AMD ROCm apt repo is
> registered. **Do not** rely on Debian's `libamdhip64-5` (HIP 5.7) — it is
> ABI-incompatible with the MIGraphX-enabled `libonnxruntime.so` (HIP 7.x) and will
> report `HipRuntimeMissing` (or, if forced, abort with an LLVM option-collision
> error). Never symlink `.so.5` → `.so.7`.

The instructions below cover the four distributions in active use. For everything
else, defer to AMD's upstream quick-start guide at
<https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html>.

### Ubuntu 22.04 / 24.04 LTS

This is the canonical install path. AMD publishes Ubuntu packages and maintains
the `amdgpu-install` script for them.

**Verify before you start:**

- `uname -r` should print `5.x.x` or `6.x.x`. Run `sudo apt-get install -y
  linux-headers-$(uname -r)` to ensure kernel headers are present (DKMS needs
  them when the amdgpu module is rebuilt against a new kernel).
- If UEFI Secure Boot is enabled in the BIOS, the DKMS module build will fail
  to sign and `amdgpu` will not auto-load. Either disable Secure Boot, or
  enroll a MOK (Machine Owner Key) — the `amdgpu-install` script will print
  instructions for the latter on the first run.
- `python3` must be on `PATH` (the `amdgpu-install` script is a Python 3 entry
  point). Ubuntu ships it by default; minimal container images do not.

**Install commands:**

```bash
# 1. Update and install prerequisites
sudo apt-get update
sudo apt-get install -y wget python3 libstdc++6

# 2. Download and run the AMD installer
#    (check https://amdgpu-install.readthedocs.io/ for the latest URL)
wget https://repo.radeon.com/amdgpu-install/6.2.3/ubuntu/jammy/amdgpu-install_6.2.3.60103-1_all.deb
sudo apt-get install -y ./amdgpu-install_6.2.3.60103-1_all.deb

# 3. Install the ROCm usecase
#    The --usecase=rocm flag pulls in hsa-rocr-dev, miopen-hip, rocblas, etc.
sudo amdgpu-install --usecase=rocm --accept-eula

# 4. Add the current user to the video and render groups
sudo usermod -aG video,render $USER

# 5. (Optional but recommended) Persist LD_LIBRARY_PATH across reboots
echo '/opt/rocm/lib' | sudo tee /etc/ld.so.conf.d/rocm.conf
sudo ldconfig
```

Log out and back in for the new group memberships to take effect. Then verify:

```bash
rocminfo | grep -E "Name:|Marketing Name:"
# Expected: at least one Agent with Name: <gfxNNNN> and Marketing Name: <GPU name>
```

**Common gotchas:**

- `amdgpu-install` fails with `Module amdgpu not found` → kernel headers are
  missing. Install `linux-headers-$(uname -r)` and retry.
- `amdgpu-install` completes but `/dev/kfd` does not appear after reboot →
  Secure Boot is blocking the DKMS-signed module. Either disable Secure Boot
  in BIOS or run `sudo mokutil --import /var/lib/dkms/mok/MOK.der` and enroll
  the key on next boot.
- `libstdc++` version too old (you see `GLIBCXX_3.4.29 not found`) → your
  Ubuntu base is older than 22.04 (e.g. 20.04 with only the security pocket
  enabled). Run `sudo apt-get install -y libstdc++6` from a `jammy` pocket, or
  upgrade to 22.04.
- `rocminfo` reports "No Agent" → the user running `rocminfo` is not in the
  `video` group. Re-run with `sudo` once to confirm the kernel side is good,
  then fix the group membership.

### Debian 12 (Bookworm)

**AMD does not officially support Debian.** This is a community-supported
configuration. The `amdgpu-install` script can be coerced into running with
`--no-dkms`, but you are accepting the risk that future ROCm releases may
break on Debian without warning.

**Verify before you start:**

- Same kernel and header checks as Ubuntu (`linux-headers-$(uname -r)`).
- Confirm `bookworm-backports` is enabled in `sources.list` if you want a
  newer kernel than the default 6.1.

**Install commands (community, unsupported):**

```bash
# 1. Prerequisites
sudo apt-get update
sudo apt-get install -y wget python3 libstdc++6

# 2. Download the Ubuntu jammy .deb — Debian is ABI-compatible for the
#    installer script itself, even though the underlying packages are
#    not in the Debian repo. You are warned.
wget https://repo.radeon.com/amdgpu-install/6.2.3/ubuntu/jammy/amdgpu-install_6.2.3.60103-1_all.deb
sudo apt-get install -y ./amdgpu-install_6.2.3.60103-1_all.deb

# 3. Run with --no-dkms because Debian's kernel packaging does not
#    trigger the DKMS hooks that amdgpu-install expects.
sudo amdgpu-install --usecase=rocm --no-dkms --accept-eula

# 4. Group membership
sudo usermod -aG video,render $USER
echo '/opt/rocm/lib' | sudo tee /etc/ld.so.conf.d/rocm.conf
sudo ldconfig
```

**Alternative: unofficial Debian ROCm packages.** The `rocm-hip-sdk` and
related packages exist in Debian's `bookworm` pocket but are several major
versions behind AMD's release. They are fine for development but lag the
library's expected runtime version by 6-12 months. Pin a version explicitly
if reproducibility matters.

**Common gotchas:**

- `amdgpu-install` fails to detect Debian and bails. Workaround: rename
  `/etc/os-release`'s `ID=ubuntu` temporarily, or use the
  `wget https://.../amdgpu-install_..._all.deb && dpkg -i` path and then
  invoke `amdgpu-install` with `--no-dkms`.
- ROCm packages pull in an Ubuntu-specific `libstdc++` that conflicts with
  Debian's slightly older version. If `apt` complains, install
  `libstdc++6` from `bookworm-backports` first.
- `--no-dkms` means the `amdgpu` module is **not** automatically rebuilt
  after a kernel upgrade. You must re-run `amdgpu-install` (or manually
  `modprobe amdgpu`) after every kernel bump.

### RHEL 9 / Rocky Linux 9 / AlmaLinux 9 / Fedora 40+

The `amdgpu-install` script supports RHEL family distros directly. SELinux
will need to be either set to permissive for the `amdgpu` domain or have a
custom policy module loaded.

**Verify before you start:**

- `dnf install -y kernel-devel kernel-headers` to ensure the build deps
  for DKMS are present.
- `getenforce` should print `Enforcing` or `Permissive`. If SELinux is
  fully disabled (`/etc/selinux/config` has `SELINUX=disabled`), the
  installer will warn but proceed.

**Install commands (RHEL 9 / Rocky 9 / Alma 9):**

```bash
# 1. Enable EPEL and CodeReady Builder (CRB) — required for some ROCm deps
sudo dnf install -y epel-release
sudo dnf config-manager --set-enabled crb   # or 'powertools' on older 9.x

# 2. Download the RHEL 9 .deb (the installer is the same; it auto-detects
#    dnf and switches package manager)
wget https://repo.radeon.com/amdgpu-install/6.2.3/rhel/9/amdgpu-install-6.2.3.60103-1.el9.noarch.rpm
sudo dnf install -y ./amdgpu-install-6.2.3.60103-1.el9.noarch.rpm

# 3. Install the ROCm usecase
sudo amdgpu-install --usecase=rocm --accept-eula

# 4. Groups and library path
sudo usermod -aG video,render $USER
echo '/opt/rocm/lib' | sudo tee /etc/ld.so.conf.d/rocm.conf
sudo ldconfig
```

**Fedora 40+:** identical to the RHEL 9 commands above but use the
`amdgpu-install-...fc40.noarch.rpm` URL from AMD's repo index.

**SELinux note:** if `rocminfo` or the inference run fails with
`Permission denied` on `/dev/kfd` despite correct group membership, audit
the denial with `sudo ausearch -m avc -ts recent` and load a custom policy
module. The faster workaround is `sudo setenforce 0` for the duration of
testing, but do not leave the system in permissive mode for production.

**Common gotchas:**

- `dnf` cannot resolve `crb` → you are on RHEL 9.0/9.1; the repo is named
  `powertools` on 9.0. Upgrade to 9.2+ or enable `powertools`.
- `amdgpu-install` exits with "SELinux is set to disabled" → it will
  proceed, but the install will not register the SELinux module. If you
  later re-enable SELinux, ROCm will be denied. Re-run the installer with
  SELinux enabled to register the module.
- DKMS build fails with `Cannot find /lib/modules/$(uname -r)/build` →
  `kernel-devel-$(uname -r)` is not installed, or you are running a
  custom kernel whose headers are in a non-standard location.

### Arch Linux

Arch has the best-in-class rolling-release ROCm support. The
`rocm-hip-sdk` meta-package pulls in everything you need at the latest
version.

**Verify before you start:**

- `pacman -Syu` to ensure the system is current. Arch's ROCm packages are
  built against the latest stable kernel; running an old kernel can cause
  module-load failures.
- For older GPUs (RDNA 1, Vega), set `HSA_OVERRIDE_GFX_VERSION` as
  described in [System Requirements](#system-requirements).

**Install commands:**

```bash
# 1. Install the ROCm meta-package
sudo pacman -S rocm-hip-sdk

# 2. Add user to groups (Arch creates the groups automatically)
sudo usermod -aG video,render $USER

# 3. Enable the amdgpu module to autoload
echo "amdgpu" | sudo tee /etc/modules-load.d/amdgpu.conf
```

Log out and back in. Verify:

```bash
rocminfo | grep -E "Name:|Marketing Name:"
```

**Common gotchas:**

- `rocm-hip-sdk` is split across `extra` and `multilib`; if you see
  "target not found" for one of the deps, enable the `multilib` repo in
  `/etc/pacman.conf`.
- Some legacy MIOpen kernels are only in the AUR. If
  `rocm-hip-sdk` installs but MIOpen fails to load a kernel, install
  `miopen-hip-gfx1030` (or the matching `gfxNNNN` for your GPU) from AUR.
- The Arch kernel is `linux` (or `linux-lts` / `linux-zen`). If you are
  running a custom kernel, the `amdgpu` module is rebuilt on install but
  may not match your running kernel. Reboot after the install.

### Other distros

For SUSE, Amazon Linux, Azure Linux, Container-Optimized OS, or any other
Linux distribution not listed above, defer to AMD's upstream quick-start:

- <https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html>

The general shape is the same: register AMD's package repo for your distro,
run `amdgpu-install --usecase=rocm`, add yourself to the `video` and
`render` groups, and verify with `rocminfo`.

For containerized deployments (Docker, Podman, Kubernetes), AMD publishes
base images at `rocm/dev-ubuntu-22.04`, `rocm/pytorch:latest`, and
similar. The `docker run` invocation must include `--device=/dev/kfd
--device=/dev/dri --group-add video --group-add render
--security-opt seccomp=unconfined` for the GPU to be visible inside the
container. The
[`docs/rocm-wsl2-runner-setup.md`](./rocm-wsl2-runner-setup.md) doc has a
worked example for Docker inside WSL2.

---

## Configuring Your Application

From the application side, enabling ROCm is a one-line change to the engine's
options.

### Minimal configuration

```csharp
using InferenceEngine.Core;
using Microsoft.Extensions.Logging;

var options = new InferenceEngineOptions
{
    UseGpuAcceleration = true,   // required for the ROCm code path to execute
    DeviceId = 0,                // GPU index; 0 for the only/primary GPU
    WarmupOnLoad = true,         // recommended for production (avoids cold-start)
};

var logger = LoggerFactory.Create(b => b.AddConsole()).CreateLogger<MyEngine>();
var engine = new MyEngine(options, logger);
await engine.LoadAsync();
```

`UseGpuAcceleration = true` is the gate. If the host does not have a usable
ROCm runtime, the engine logs "ROCm runtime not available on this host —
using CPU provider" and runs CPU-only. If the host has ROCm but
`UseGpuAcceleration = false`, the engine skips the runtime check entirely
and goes straight to CPU. The default value of `UseGpuAcceleration` is
`true` (see [`InferenceEngineOptions.cs`](../InferenceEngine.Core/InferenceEngineOptions.cs)),
so an `InferenceEngineOptions` constructed with no overrides is already
GPU-attempting.

### Behavior truth table

The runtime behavior for each combination of `UseGpuAcceleration` and host
capability:

| `UseGpuAcceleration` | Host ROCm/MIGraphX stack | Behavior | `EngineInfo.Provider` |
|---|---|---|---|
| `true` | `/dev/kfd` + HSA + HIP 7.x + MIGraphX all present | MIGraphX execution provider appended | `MIGraphX` |
| `true` | A layer is missing (e.g. `libamdhip64.so.7`) | CPU fallback + a precise `FallbackReason` | `Cpu` (`IsFallback=true`) |
| `false` | (not checked) | Unconditional CPU | `Cpu` |

`EngineInfo.Provider` (with `EngineInfo.IsFallback` and `EngineInfo.FallbackReason`)
is the single source of truth at runtime. Read it after `LoadAsync` returns — or
receive it asynchronously via the `InferenceEngineOptions.OnSessionInitialized`
callback.

### Confirming the EP at runtime

```csharp
var engine = new MyEngine(options, logger);
await engine.LoadAsync();

if (engine.EngineInfo.Provider == InferenceProvider.MIGraphX)
{
    Console.WriteLine($"MIGraphX EP active on device {engine.EngineInfo.DeviceId}.");
}
else
{
    Console.WriteLine($"CPU fallback — reason: {engine.EngineInfo.FallbackReason}.");
}
```

For the full ordered diagnostic trail (every Pass/Fail/Skip during provider
selection), iterate `engine.EngineInfo.Checks`. This is the pattern the test suite
asserts on — see `EngineInfo_LinuxX64WithKfd_*` in
[`EngineInfoTests.cs`](../InferenceEngine.Tests/EngineInfoTests.cs).

### Why the engine never throws on a missing GPU

A common concern: "if my CI runs on a non-ROCm host, will the engine crash?"
No. The provider-selection probe records the first failing check (missing
`/dev/kfd`, unresolvable HSA/HIP/MIGraphX runtime, or any exception during a
`NativeLibrary.TryLoad`) and falls back to CPU with a precise
`InferenceFallbackReason` — it never throws. The test suite uses
`Xunit.SkippableFact` with a host-capability skip precondition to skip cleanly on
non-GPU hosts rather than failing — see the `EngineInfo_LinuxX64WithKfd_*` tests in
[`EngineInfoTests.cs`](../InferenceEngine.Tests/EngineInfoTests.cs).

If you want a hard failure on a non-GPU host (e.g. for a production deployment that
requires GPU), assert on `EngineInfo.Provider` after `LoadAsync` and throw if it is
`Cpu`.

---

## Troubleshooting

The library's design is "auto-detect and log, never crash on missing GPU."
Every row below describes a log message you might see, the most likely cause,
and the fix. Issues are ordered from "most common on first-time setup" to
"exotic scenarios."

| Symptom | Likely cause | Fix |
|---|---|---|
| Engine logs `ROCm runtime not available on this host — using CPU provider`; `EngineInfo.Provider == Cpu` | Either `/dev/kfd` is missing, or `libhsa-runtime64.so` is not in the dynamic linker's search path. The two checks in `IsRocmRuntimeUsable()` are independent — you can have one without the other. | (1) `ls -la /dev/kfd` — if missing, re-check the `amdgpu` module is loaded (`lsmod \| grep amdgpu`) and the kernel is recent. (2) `ldconfig -p \| grep libhsa-runtime64` — if empty, install `hsa-rocr-dev` (or distro equivalent) and run `echo /opt/rocm/lib \| sudo tee /etc/ld.so.conf.d/rocm.conf && sudo ldconfig`. (3) As a last resort, set `LD_LIBRARY_PATH=/opt/rocm/lib` in the application environment. |
| Engine throws `OnnxRuntimeException` (not the recoverable native-load kind) after the auto-detect passes; the message references `HSA`, `hip`, or `MIOpen` | One of the downstream ROCm libraries (`libamdhip64.so`, `libmiopen.so`, `librocblas.so`) is missing or built for a different ROCm version than the `InferenceEngine.ROCm.Runtime.linux-x64` package was compiled against. A wrong GFX version is a sub-case. | (1) `ldconfig -p \| grep -E 'libamdhip64\|libmiopen\|librocblas'` — every line should be present. (2) `rocminfo` should run cleanly and show at least one Agent. (3) If the GPU is on the ROCm support list but `rocminfo` reports "No devices found", set `HSA_OVERRIDE_GFX_VERSION=10.3.0` (for RDNA 1 / Navi 10) or the matching version for your GPU family. (4) Check `dmesg \| grep -i amdgpu` for driver-side errors. (5) The two `InferenceEngine.ROCm.Runtime.linux-x64` and the `amdgpu-install` `--usecase=rocm` packages must be a coherent version pair; mixing ROCm 6.x drivers with ROCm 5.x userspace (or vice versa) is a common source of `hipErrorNoBinaryForGPU`. |
| `rocminfo` shows the GPU correctly, but the library reports ROCm unavailable | The `rocminfo` binary is statically linked or has `/opt/rocm/lib` baked in via its own RPATH, so it finds the HSA runtime even when `ldconfig` does not. The library, on the other hand, uses the system `dlopen` search path, which follows `LD_LIBRARY_PATH` and `ldconfig`. | Set `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64` in the application's environment, or persist it system-wide with the `ld.so.conf.d` snippet from Step 5 above. Restart the application to pick up the new path. |
| Engine logs `Permission denied` opening `/dev/kfd` despite running as a non-root user | User is not in the `video` and `render` groups, or the groups exist but the user was not added (`/etc/group` shows the user is missing). This is the most common "everything is installed and it still doesn't work" symptom. | `sudo usermod -aG video,render $USER` and **log out and back in** (or `newgrp video render` in the current shell — but a full re-login is more reliable). Verify with `id $USER` — both groups should be listed. Verify device access with `ls -la /dev/kfd /dev/dri/renderD128` — the group should be `video` or `render` and the user should be a member. |
| Engine produces correct predictions and `EngineInfo.Provider == MIGraphX`, but inference latency is the same as CPU (or only marginally better) | ROCm is initialized, but MIOpen is silently falling back to CPU implementations of the conv kernels. This is common for RDNA 1 GPUs, older kernels with no native GPU codepath, or models that hit a non-fused op (e.g. a layout-unsupported `Reshape`). | (1) Set `MIOPEN_LOG_LEVEL=3` in the application environment and re-run; you will see one line per conv op identifying which algorithm was selected and whether it is `MIOpen` (GPU) or `fft/fft-direct/winograd` (CPU). (2) If most ops are CPU, set `HSA_ENABLE_SDMA=0` to force DMA paths and re-test. (3) For older GPUs, `HSA_OVERRIDE_GFX_VERSION=10.3.0` (RDNA 1) or `9.0.0` (Vega) unlocks the GPU codepaths the kernel would otherwise refuse to compile. (4) Check the ONNX model's opset — opset 13+ is recommended; very old opsets may emit `Reshape` patterns that MIOpen has no GPU kernel for. (5) `rocm-profile` or `rocprof` can attribute per-op time; if the bottleneck is a single op, profile and consider a model surgery. |
| The library's post-Build `RestoreUpstreamCpuLibOnLinux` target did not fire, and the published application ships the ROCm-enabled `libonnxruntime.so` (the one that SIGSEGVs on non-ROCm hosts) | The build was not invoked with `--runtime linux-x64` — the target is gated on `'$(RuntimeIdentifier)' == 'linux-x64'`. A no-RID build (`dotnet build` without `-r`) silently drops the target because `RuntimeIdentifier` is empty. Custom NuGet cache locations in CI can also break the `$(NuGetPackageRoot)microsoft.ml.onnxruntime/1.24.1/...` path the target depends on. | (1) Verify with `dotnet --info` that the RID graph is enabled (the project sets `<UseRidGraph>true</UseRidGraph>`) and that you are building with `-r linux-x64` or `--runtime linux-x64`. (2) Check the build output: `dotnet build -r linux-x64 -v:n` should show `Target "RestoreUpstreamCpuLibOnLinux"` in the `AfterTargets="Build"` chain. (3) If `NuGetPackageRoot` is non-default, the path in the target is wrong; you can override `NuGetPackageRoot` on the build command line and the target will pick it up via MSBuild property evaluation. (4) Confirm the output dir's `libonnxruntime.so` is the CPU build (run `file bin/Debug/net10.0/linux-x64/libonnxruntime.so` — it should NOT say "ROCm" in the build ID). |
| Engine logs `DirectML failed to load` or `CoreML failed to load` on Linux | A derived engine has been written to dispatch to all three execution providers unconditionally. On Linux, only the ROCm path is reached. | Confirm `ConfiguredSessionOptions` in your engine's base class is `BaseInferenceEngine.ConfiguredSessionOptions` (i.e. you did not override it). If you must override it, preserve the platform branching — Linux x64 → ROCm (with the runtime check), Linux ARM64 → CPU, Windows → DirectML, macOS → CoreML. |
| Application segfaults on first call to `engine.PredictAsync` (no managed exception, just process termination) | The `InferenceEngine.ROCm.Runtime.linux-x64` package's `libonnxruntime.so` is being loaded but the host has no `/dev/kfd`. This is a known issue with the dev release; the post-Build `RestoreUpstreamCpuLibOnLinux` target is meant to prevent it but only fires on `linux-x64` RID builds. | Confirm you built with `--runtime linux-x64` (not just `linux`). Confirm the output dir's `libonnxruntime.so` is the upstream CPU build, not the ROCm dev one — `sha256sum` the file and compare with the upstream package's `libonnxruntime.so` for the matching version. If you are on a no-RID build, the `RestoreUpstreamCpuLibOnLinux` target is gated off and the ROCm `.so` is in play; the SIGSEGV on non-ROCm hosts is a known limitation. |
| Engine logs `Native session initialization failed. Falling back to CPU.` at warning level, then succeeds on CPU | The catch in `LoadAsync` matched `IsRecoverableNativeFailure` and fell back. The wrapped exception will be in the log (it is logged at warning level before the fallback `SessionOptions` is created). | Inspect the warning's `Exception` payload. The two recoverable cases are: (a) `libonnxruntime.so` itself is missing or unloadable (check the output dir with `ls -la bin/.../linux-x64/`), or (b) a TypeInitializationException wrapping one of those. (c) The `RestoreUpstreamCpuLibOnLinux` target may not have run — see the row above. |
| ROCm works on a fresh install but breaks after a kernel upgrade | The `amdgpu` DKMS module was not rebuilt for the new kernel, so the old module is unloadable. The `dmesg` log will show `disagrees about version of symbol module_layout`. | Install matching kernel headers (`sudo apt-get install linux-headers-$(uname -r)` on Ubuntu, `sudo dnf install kernel-devel-$(uname -r)` on RHEL family), then `sudo dpkg-reconfigure amdgpu-dkms` (Ubuntu) or `sudo dkms autoinstall` (RHEL) to trigger a rebuild. Reboot. |

---

## References

Upstream documentation that informed the design and the verification steps in
this guide:

- **AMD ROCm installation guide** — <https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html>
- **AMD GPU support matrix (Linux)** — <https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html>
- **AMD `amdgpu-install` reference** — <https://amdgpu-install.readthedocs.io/en/latest/>
- **ONNX Runtime ROCm execution provider** — <https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html>
- **HSA runtime programming guide** — <https://rocm.docs.amd.com/projects/HSA-Runtime-Reference/>
- **MIOpen environment variables and logging** — <https://rocm.docs.amd.com/projects/MIOpen/>

In-repo documentation referenced by this guide:

- [`docs/Review___ONNX_Runtime_Missing_Native_RESOLUTION.md`](./Review___ONNX_Runtime_Missing_Native_RESOLUTION.md) —
  the design and validation record for the Linux native-library fixes
  (Delete + Symlink + Restore post-Build targets) and the runtime check in
  `BaseInferenceEngine.IsRocmRuntimeUsable()`. Read this if you are debugging
  a `DllNotFoundException` or a SIGSEGV in `libonnxruntime.so`.
- [`docs/rocm-wsl2-runner-setup.md`](./rocm-wsl2-runner-setup.md) — a worked
  example for setting up a GitHub Actions self-hosted runner inside WSL2 with
  AMD GPU passthrough. Useful as a reference for any non-bare-metal Linux host
  (WSL2, Hyper-V with GPU paravirtualization, KVM with GPU passthrough).
- [`docs/inferenceapp-integration-guide.md`](./inferenceapp-integration-guide.md) —
  end-to-end integration of an `IInferenceEngine<TInput, TOutput>` consumer
  into a real application, including the GPU configuration knobs.
- [`AGENTS.md`](../AGENTS.md) — repository conventions, build/test commands,
  and the `agent-self-check.sh` workflow used for AI-authored changes.
- [`InferenceEngine.Core/BaseInferenceEngine.cs`](../InferenceEngine.Core/BaseInferenceEngine.cs) —
  the source of truth for `IsRocmRuntimeUsable()` and the `ConfiguredSessionOptions`
  platform-dispatch logic. Lines 237–324 are the relevant block.
- [`InferenceEngine.Core/InferenceEngine.Core.csproj`](../InferenceEngine.Core/InferenceEngine.Core.csproj) —
  the post-Build targets (`RemoveWindowsNativeLibsOnNonWindows`,
  `SymlinkLinuxNativeLibsForPInvoke`, `RestoreUpstreamCpuLibOnLinux`) and the
  `InferenceEngine.ROCm.Runtime.linux-x64` `PackageReference` (lines 62–143).
- [`InferenceEngine.Core/InferenceEngineOptions.cs`](../InferenceEngine.Core/InferenceEngineOptions.cs) —
  the `UseGpuAcceleration`, `DeviceId`, and `WarmupOnLoad` options and their
  defaults.
