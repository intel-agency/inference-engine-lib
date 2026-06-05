# Linux ROCm Configuration

## Overview

This guide covers enabling GPU hardware acceleration for ONNX Runtime inference on
Linux x64 hosts that have an AMD GPU installed, using AMD's ROCm execution provider.
The library auto-detects ROCm at process start and falls back to CPU silently on
hosts where the kernel fusion driver, the HSA userspace runtime, or the
`libonnxruntime_providers_rocm.so` provider plugin is missing. The same auto-fallback
shape is used on Windows (DirectML) and macOS (CoreML) — a developer that ships a
cross-platform agent does not need to fork their engine code per OS.

For the design rationale and the full set of native-library fixes that made the
runtime check work correctly on Linux, see
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

The minimum set of userspace packages the library's runtime check actually needs is
narrow:

- `libhsa-runtime64.so` (provided by `hsa-rocr-dev` or distro equivalent). This is
  the single library the probe in `BaseInferenceEngine.IsRocmRuntimeUsable()` calls
  `NativeLibrary.Load` against.
- `libhsa-runtime64.so.1` (the SONAME link to the above) in the dynamic linker's
  search path, i.e. discoverable via `ldconfig` or `LD_LIBRARY_PATH`.
- For actual GPU work (not just the probe), the engine will also load
  `libonnxruntime_providers_rocm.so` and that provider transitively needs
  `libhsa-runtime64.so`, `libamdhip64.so`, `libmiopen.so`, and `librocblas.so` at
  the same paths. These are pulled in by the larger `rocm` meta-packages and are
  not individually required at the probe stage.

Canonical install method: AMD's `amdgpu-install` script. It detects the running
distro and registers AMD's package repositories with the right GPG keys, then
installs a coherent set of ROCm packages at a known-good version. Distro packages
(e.g. Debian's `rocm-hip-sdk`, Arch's `rocm-hip-sdk`) are usable but lag the AMD
release by weeks to months; they are fine for development, less so for production
where you want a specific ROCm version pinned.

Environment variables that frequently need to be set when running the application:

- `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64` (some ROCm installs only put
  libraries in `/opt/rocm/lib`, some only in `/opt/rocm/lib64`; setting both is
  harmless and covers the union). Modern `.deb`/`.rpm` installs register
  `/etc/ld.so.conf.d/rocm.conf` and run `ldconfig` automatically, so this is usually
  a no-op but is the first thing to check if the runtime probe fails.
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
  `1.19.2-dev.28`) is referenced unconditionally on `linux-x64` builds. It ships a
  ROCm-enabled `libonnxruntime.so` plus the
  `libonnxruntime_providers_rocm.so` provider plugin.
- A post-Build MSBuild target named `RestoreUpstreamCpuLibOnLinux` runs after the
  NuGet restore and copies the upstream `Microsoft.ML.OnnxRuntime` package's
  CPU-only `libonnxruntime.so` over the ROCm-enabled one in the output directory.
  This means the `.so` that the P/Invoke layer actually loads on every Linux x64
  build is the safe, CPU-only one; the ROCm provider `.so` is still in the output
  dir and gets loaded on demand at runtime.
- The .NET 10 P/Invoke search algorithm on Linux probes for `libonnxruntime.dll.so`,
  `onnxruntime.dll.so`, `libonnxruntime.dll`, and `onnxruntime.dll` (in that order).
  It does **not** probe for `libonnxruntime.so` (the file the package actually
  ships). A second post-Build target named `SymlinkLinuxNativeLibsForPInvoke`
  creates the required symlinks so the P/Invoke name resolution finds the
  library on disk.
- The runtime check (`BaseInferenceEngine.IsRocmRuntimeUsable()`) is the
  decision point. It does two things in order:
  1. Verifies that `/dev/kfd` exists on the filesystem.
  2. Calls `NativeLibrary.Load("libhsa-runtime64.so", ...)` to confirm the HSA
     userspace runtime is resolvable.
  If both succeed, the ROCm execution provider is appended to the session
  options. If either fails, the engine logs at info level and falls back to CPU.

For the full timeline of the native-library fixes (the original 5 review comments
plus the 4 additional root-cause fixes for upstream package quirks, SkiaSharp
split packages, and the SIGSEGV-on-non-ROCm-hosts hazard), see
[`docs/Review___ONNX_Runtime_Missing_Native_RESOLUTION.md`](./Review___ONNX_Runtime_Missing_Native_RESOLUTION.md).

---

## Verifying Your System

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

The runtime check in `BaseInferenceEngine.IsRocmRuntimeUsable()` calls
`NativeLibrary.Load("libhsa-runtime64.so", ...)`. The library must be
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

Then in a small C# test program, or via the example `FaceDetectionEngine`:

```text
Loading...
Platform detected: Linux x64. ROCm runtime detected. Attempting ROCm provider...
Model loaded. Primary Input: images
Warmup complete.
IsRocmSessionActive = True
```

A successful ROCm path produces the line **"ROCm runtime detected. Attempting
ROCm provider..."** followed by `IsRocmSessionActive == True`.

A fallback path (which is **not an error** — it is by design) produces:

```text
Platform detected: Linux x64. ROCm runtime not available on this host — using CPU provider.
Model loaded. Primary Input: images
IsRocmSessionActive = False
```

If you see the fallback line, re-run Steps 3-6 and fix the first one that
fails. The library will not re-attempt ROCm on a later call — the decision is
made once at `LoadAsync` time.

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

| `UseGpuAcceleration` | Host has `/dev/kfd` + `libhsa-runtime64.so` | Behavior | `IsRocmSessionActive` |
|---|---|---|---|
| `true` | Yes | ROCm execution provider appended | `true` |
| `true` | No | Info log + CPU fallback | `false` |
| `false` | (not checked) | Unconditional CPU | `false` |
| `true` | Yes, but `AppendExecutionProvider_ROCm` throws | Warning log + CPU fallback (caught in the `ConfiguredSessionOptions` try block) | `false` |
| `true` | Yes, but a downstream native lib (`libamdhip64.so`, `libmiopen.so`, `librocblas.so`) is missing | `OnnxRuntimeException` at session creation; outer catch with `IsRecoverableNativeFailure` may catch it and fall back to CPU, depending on the exception type. If uncaught, it propagates to the `LogError` handler. | `false` if recovered; session creation throws otherwise |

The `IsRocmSessionActive` flag is the single source of truth at runtime.
Read it after `LoadAsync` returns to confirm which path was actually used.

### Confirming the EP at runtime

```csharp
var engine = new MyEngine(options, logger);
await engine.LoadAsync();

if (engine.IsRocmSessionActive)
{
    Console.WriteLine("ROCm EP active — running on AMD GPU.");
}
else
{
    Console.WriteLine("CPU fallback — ROCm not available on this host.");
}
```

This is also the pattern the test suite uses — see
[`FaceDetectionEngineTests.PredictAsync_OnLinuxX64WithAmdGpu_ShouldUseRocmProvider`](../InferenceEngine.Tests/FaceDetectionEngineTests.cs)
for the canonical example with full skip-precondition chaining.

### Why the engine never throws on a missing GPU

A common concern: "if my CI runs on a non-ROCm host, will the engine crash?"
No. The runtime check in `IsRocmRuntimeUsable()` returns `false` on any
failure (missing `/dev/kfd`, unresolvable HSA runtime, or any exception
during the `NativeLibrary.Load` probe), and the engine falls back to CPU.
The test in
[`FaceDetectionEngineTests.cs`](../InferenceEngine.Tests/FaceDetectionEngineTests.cs)
uses `Skip.IfNot(IsRocmRuntimeUsableOnHost(), ...)` to skip cleanly on
non-ROCm hosts rather than failing.

If you want a hard failure on a non-ROCm host (e.g. for a production
deployment that requires GPU), assert on `IsRocmSessionActive` after
`LoadAsync` and throw if it is `false`.

---

## Troubleshooting

The library's design is "auto-detect and log, never crash on missing GPU."
Every row below describes a log message you might see, the most likely cause,
and the fix. Issues are ordered from "most common on first-time setup" to
"exotic scenarios."

| Symptom | Likely cause | Fix |
|---|---|---|
| Engine logs `ROCm runtime not available on this host — using CPU provider`; `IsRocmSessionActive == false` | Either `/dev/kfd` is missing, or `libhsa-runtime64.so` is not in the dynamic linker's search path. The two checks in `IsRocmRuntimeUsable()` are independent — you can have one without the other. | (1) `ls -la /dev/kfd` — if missing, re-check the `amdgpu` module is loaded (`lsmod \| grep amdgpu`) and the kernel is recent. (2) `ldconfig -p \| grep libhsa-runtime64` — if empty, install `hsa-rocr-dev` (or distro equivalent) and run `echo /opt/rocm/lib \| sudo tee /etc/ld.so.conf.d/rocm.conf && sudo ldconfig`. (3) As a last resort, set `LD_LIBRARY_PATH=/opt/rocm/lib` in the application environment. |
| Engine throws `OnnxRuntimeException` (not the recoverable native-load kind) after the auto-detect passes; the message references `HSA`, `hip`, or `MIOpen` | One of the downstream ROCm libraries (`libamdhip64.so`, `libmiopen.so`, `librocblas.so`) is missing or built for a different ROCm version than the `InferenceEngine.ROCm.Runtime.linux-x64` package was compiled against. A wrong GFX version is a sub-case. | (1) `ldconfig -p \| grep -E 'libamdhip64\|libmiopen\|librocblas'` — every line should be present. (2) `rocminfo` should run cleanly and show at least one Agent. (3) If the GPU is on the ROCm support list but `rocminfo` reports "No devices found", set `HSA_OVERRIDE_GFX_VERSION=10.3.0` (for RDNA 1 / Navi 10) or the matching version for your GPU family. (4) Check `dmesg \| grep -i amdgpu` for driver-side errors. (5) The two `InferenceEngine.ROCm.Runtime.linux-x64` and the `amdgpu-install` `--usecase=rocm` packages must be a coherent version pair; mixing ROCm 6.x drivers with ROCm 5.x userspace (or vice versa) is a common source of `hipErrorNoBinaryForGPU`. |
| `rocminfo` shows the GPU correctly, but the library reports ROCm unavailable | The `rocminfo` binary is statically linked or has `/opt/rocm/lib` baked in via its own RPATH, so it finds the HSA runtime even when `ldconfig` does not. The library, on the other hand, uses the system `dlopen` search path, which follows `LD_LIBRARY_PATH` and `ldconfig`. | Set `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64` in the application's environment, or persist it system-wide with the `ld.so.conf.d` snippet from Step 5 above. Restart the application to pick up the new path. |
| Engine logs `Permission denied` opening `/dev/kfd` despite running as a non-root user | User is not in the `video` and `render` groups, or the groups exist but the user was not added (`/etc/group` shows the user is missing). This is the most common "everything is installed and it still doesn't work" symptom. | `sudo usermod -aG video,render $USER` and **log out and back in** (or `newgrp video render` in the current shell — but a full re-login is more reliable). Verify with `id $USER` — both groups should be listed. Verify device access with `ls -la /dev/kfd /dev/dri/renderD128` — the group should be `video` or `render` and the user should be a member. |
| Engine produces correct predictions and `IsRocmSessionActive == true`, but inference latency is the same as CPU (or only marginally better) | ROCm is initialized, but MIOpen is silently falling back to CPU implementations of the conv kernels. This is common for RDNA 1 GPUs, older kernels with no native GPU codepath, or models that hit a non-fused op (e.g. a layout-unsupported `Reshape`). | (1) Set `MIOPEN_LOG_LEVEL=3` in the application environment and re-run; you will see one line per conv op identifying which algorithm was selected and whether it is `MIOpen` (GPU) or `fft/fft-direct/winograd` (CPU). (2) If most ops are CPU, set `HSA_ENABLE_SDMA=0` to force DMA paths and re-test. (3) For older GPUs, `HSA_OVERRIDE_GFX_VERSION=10.3.0` (RDNA 1) or `9.0.0` (Vega) unlocks the GPU codepaths the kernel would otherwise refuse to compile. (4) Check the ONNX model's opset — opset 13+ is recommended; very old opsets may emit `Reshape` patterns that MIOpen has no GPU kernel for. (5) `rocm-profile` or `rocprof` can attribute per-op time; if the bottleneck is a single op, profile and consider a model surgery. |
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
