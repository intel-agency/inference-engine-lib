# ROCm Self-Hosted Runner Setup — PRECISION5820 (RX 6700 XT)

**Target machine:** Dell PRECISION5820, Windows 11, AMD Radeon RX 6700 XT (`gfx1031`)  
**Goal:** Register a GitHub Actions self-hosted runner with labels `self-hosted,linux,x64,amd-gpu`
inside a WSL2 Ubuntu environment, so the `rocm-gpu-test` CI workflow can execute
end-to-end ROCm inference tests against real AMD GPU hardware.

> **IMPORTANT — GPU access requires a live Windows desktop session.**
> The AMD Windows GPU driver initializes the ROCm compute interface (`/dev/kfd` inside WSL2)
> only when a user is actively logged in on the physical console or via a remote session
> where the GPU is initialized. Remote Desktop (RDP) sessions typically do **not** initialize
> the GPU. You must either:
>
> - Be physically present at the machine, or
> - Use a persistent remote session tool that keeps the desktop session alive (e.g. TeamViewer,
>   Parsec, or "Console RDP" via `mstsc /admin` with the session kept active).
>
>
> A headless / locked / no-user-session state will result in `/dev/kfd` not appearing in WSL2.

---

## Prerequisites

Before starting, confirm on the Windows host:

- Windows 11 (22H2 or later) — required for WSL2 GPU support
- AMD Radeon RX 6700 XT physically installed and visible in Device Manager
- Internet access for package downloads
- PowerShell 7+ available (`winget` is also helpful)
- A GitHub PAT or the runner registration token for `intel-agency/inference-engine-lib`
  (Settings → Actions → Runners → New self-hosted runner → copy the token)

---

## Phase 1 — Windows: Install AMD HIP SDK

The standard Adrenalin gaming driver does **not** expose `/dev/kfd` into WSL2.
The **HIP SDK for Windows** (formerly "ROCm for Windows") includes the WSL2 bridge.

1. Open a browser and go to: <https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html>
2. Download the **HIP SDK** installer (select Windows, latest stable version — 6.x recommended).
3. Run the installer as Administrator. Select:
   - [x] HIP SDK Runtime
   - [x] HIP SDK Development Tools
   - [x] AMD ROCm (WSL) — **this checkbox is what matters**
4. Reboot Windows when prompted.
5. After reboot, confirm the driver is active:

```powershell
# Run in PowerShell (Windows)
Get-PnpDevice -FriendlyName "*Radeon*" | Select-Object Status, FriendlyName
```

Expected: `Status = OK`.

---

## Phase 2 — Windows: Enable WSL2

Run in an **elevated** PowerShell terminal:

```powershell
# Enable WSL2 features
wsl --install -d Ubuntu-22.04

# If WSL is already installed but Ubuntu is not:
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04
```

Reboot if prompted. WSL2 must be using the virtual machine platform (not WSL1).

Verify after reboot:

```powershell
wsl --list --verbose
# Should show Ubuntu-22.04 with VERSION=2 and STATE=Running or Stopped
```

---

## Phase 3 — WSL2: Verify GPU Passthrough

Open a WSL2 Ubuntu terminal:

```bash
wsl -d Ubuntu-22.04
```

Inside WSL2:

```bash
# Check that AMD GPU devices are exposed
ls -la /dev/kfd /dev/dri/render*
```

Expected output (may vary by render node number):

```text
crw-rw---- 1 root render 234, 0 Apr  7 12:00 /dev/kfd
crw-rw-rw- 1 root render 226, 128 Apr  7 12:00 /dev/dri/renderD128
```

> If `/dev/kfd` is **missing**, the AMD HIP SDK WSL component did not install correctly
> or the Windows session is not active with GPU access. Reboot Windows and ensure
> you are logged in on the desktop before proceeding.

---

## Phase 4 — WSL2: Install Docker Engine

Do **not** use Docker Desktop. Install Docker Engine directly inside WSL2.

```bash
# Inside WSL2 Ubuntu-22.04

# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker apt repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

# Add current user to required groups
sudo usermod -aG docker $USER
sudo usermod -aG video $USER
sudo usermod -aG render $USER

# Start Docker daemon
sudo service docker start
```

Log out and back in to WSL2 (or `newgrp docker`) for group changes to take effect.

Verify Docker is working:

```bash
docker run --rm hello-world
```

---

## Phase 5 — WSL2: Verify GPU Inside a Docker Container

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  rocm/dev-ubuntu-22.04:6.0.2 \
  rocminfo
```

Expected output includes a section like:

```text
*******                  
Agent 2                  
*******                  
  Name:                    gfx1031
  ...
  Device Type:             GPU
```

`gfx1031` is the architecture for the RX 6700 XT — this confirms GPU passthrough is working.

---

## Phase 6 — WSL2: Install .NET 10 SDK

The GitHub Actions runner and tests require .NET 10.

```bash
# Install Microsoft package repository
wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb

# Install .NET 10 SDK
sudo apt-get update
sudo apt-get install -y dotnet-sdk-10.0

# Verify
dotnet --version
# Should print 10.x.x
```

---

## Phase 7 — WSL2: Register the GitHub Actions Self-Hosted Runner

> Get the runner registration token from:
> <https://github.com/intel-agency/inference-engine-lib/settings/actions/runners/new>
> (requires repo admin access — select Linux / x64, copy the token from the `--token` line)

```bash
# Inside WSL2 — create a dedicated directory
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download the latest runner package (check https://github.com/actions/runner/releases for latest version)
RUNNER_VERSION=2.323.0
curl -o actions-runner-linux-x64.tar.gz -L \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"

tar xzf ./actions-runner-linux-x64.tar.gz

# Configure the runner — replace <TOKEN> with the token from GitHub
./config.sh \
  --url https://github.com/intel-agency/inference-engine-lib \
  --token <TOKEN> \
  --name "precision5820-wsl2-amd-gpu" \
  --labels "self-hosted,linux,x64,amd-gpu" \
  --work "_work" \
  --unattended
```

---

## Phase 8 — WSL2: Run the Runner as a Background Service

WSL2 doesn't support `systemd` by default on Ubuntu 22.04. Use a simple startup approach:

**Option A — Enable systemd in WSL2** (recommended, Ubuntu 22.04 supports it):

```bash
# Check if systemd is running
ps -p 1 -o comm=
# If output is "systemd", it's already enabled — skip the config step below

# If not enabled, add to /etc/wsl.conf:
sudo tee /etc/wsl.conf > /dev/null <<'EOF'
[boot]
systemd=true
EOF

# Exit WSL2, then from PowerShell:
#   wsl --shutdown
#   wsl -d Ubuntu-22.04
# Re-enter WSL2 and verify: ps -p 1 -o comm= (should print "systemd")
```

Then install the runner as a systemd service:

```bash
cd ~/actions-runner

# Install service (prompts for sudo)
sudo ./svc.sh install

# Start it
sudo ./svc.sh start

# Check status
sudo ./svc.sh status
```

**Option B — tmux session** (simpler, survives WSL2 session, but not reboots):

```bash
sudo apt-get install -y tmux
tmux new-session -d -s runner "cd ~/actions-runner && ./run.sh"

# Verify it is running
tmux ls
```

For a persistent-across-reboot Option B, add to WSL2's `~/.bashrc`:

```bash
# Auto-start runner in tmux if not already running
if ! tmux has-session -t runner 2>/dev/null; then
  tmux new-session -d -s runner "cd ~/actions-runner && ./run.sh"
fi
```

---

## Phase 9 — Verify the Runner Appears in GitHub

1. Go to: <https://github.com/intel-agency/inference-engine-lib/settings/actions/runners>
2. The runner `precision5820-wsl2-amd-gpu` should appear with status **Idle**.
3. Labels should show: `self-hosted`, `linux`, `x64`, `amd-gpu`.

---

## Phase 10 — Trigger a Test Run

From PowerShell on any machine with the GitHub CLI installed:

```powershell
gh workflow run rocm-gpu-test.yml --repo intel-agency/inference-engine-lib
```

Or manually via: <https://github.com/intel-agency/inference-engine-lib/actions/workflows/rocm-gpu-test.yml>
→ "Run workflow" button.

Monitor the run:

```powershell
gh run list --limit 5 --repo intel-agency/inference-engine-lib
gh run watch <run-id> --exit-status --repo intel-agency/inference-engine-lib
```

A successful run will show:

- `Verify AMD GPU available` step: prints `/dev/kfd present`
- `Run ROCm GPU inference tests` step: `PredictAsync_OnLinuxX64WithAmdGpu_ShouldUseRocmProvider` PASSES (not skipped)

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `/dev/kfd` missing in WSL2 | HIP SDK WSL component not installed, or no active Windows desktop session | Reinstall HIP SDK, ensure user is logged in on Windows desktop |
| `rocminfo` shows no GPU agents | Driver version mismatch | Ensure HIP SDK version matches; try `6.0.x` |
| Docker: `permission denied /dev/kfd` | User not in `render`/`video` groups | `sudo usermod -aG render,video $USER` then re-login |
| `gfx1031` not recognized by container image | Container image compiled for different gfx targets | The inference-engine-rocm build targets `gfx1031` explicitly — use that package version |
| Runner shows offline in GitHub | WSL2 session ended | Restart WSL2 and re-run `./run.sh` (or enable systemd service) |
| `hipErrorNoBinaryForGPU` | Wrong ONNX Runtime binary for this GPU arch | Ensure using `InferenceEngine.ROCm.Runtime.linux-x64` v1.19.2-dev.28+ which targets gfx1031 |

---

## Notes on Windows Session Requirement

The AMD ROCm compute stack on Windows works through the **WDDM kernel-mode driver**.
Unlike NVIDIA (which has a separate headless compute path), AMD's WDDM path requires
an active Windows display session to initialize the GPU for compute. This means:

- **Physical console session** → works reliably
- **Virtual display driver (VDD) + auto-login** → works reliably (**recommended for always-on unattended use**)
- **RDP session that becomes the console session** (`mstsc /admin` or `/console`) → may work
- **Standard RDP session** (non-console) → typically does **not** expose `/dev/kfd` in WSL2
- **Headless / no user logged in** → **does not work**

---

## Recommended: Virtual Display Driver (VDD) + Auto-Login for Always-On GPU Access

The part that keeps `/dev/kfd` active is purely the **virtual display** — it makes Windows
believe a monitor is attached, which causes the WDDM driver to fully initialize the GPU.
No remote-access application is needed.

The recommended VDD project is **VirtualDrivers/Virtual-Display-Driver** (MIT license,
8.9k stars, actively maintained, installable via winget):
<https://github.com/VirtualDrivers/Virtual-Display-Driver>

Several other IDD projects exist (parsec-vdd, virtual-display-rs, RustDeskIddDriver, etc.) —
see the [comparison table](https://github.com/VirtualDrivers/Virtual-Display-Driver?tab=readme-ov-file#-comparison-with-other-idds)
for a feature breakdown. VirtualDrivers/VDD is the most featureful and actively maintained option.

Combined with Windows auto-login (so there is an active user session), `/dev/kfd` is
exposed into WSL2 as long as the machine is powered on.

### Setup

1. Install the VDD via winget (simplest):

   ```powershell
   winget install --id=VirtualDrivers.Virtual-Display-Driver -e
   ```

   Or download the **Virtual Driver Control (VDC)** installer manually from the
   [releases page](https://github.com/VirtualDrivers/Virtual-Display-Driver/releases)
   and run it on the PRECISION5820.

2. Launch the **Virtual Driver Control** (VDC) app, click **Install**, then use it to
   **add a virtual display** at your preferred resolution (e.g. 1920×1080). The VDD is
   now presenting a synthetic monitor to Windows, which keeps the GPU fully initialized.

   > **Note:** If you plan to update AMD GPU or chipset drivers in future, uninstall VDD
   > first via VDC, complete the driver update, then reinstall — otherwise Windows may
   > prioritize the virtual display and boot to a black screen.

3. Configure Windows 11 **auto-login** so a user session exists after every reboot:

   ```powershell
   # Download Sysinternals Autologon (stores credentials encrypted in the registry)
   winget install Microsoft.Sysinternals.AutoLogon
   # Run Autologon.exe, enter credentials, click Enable
   ```

4. Reboot the machine. After reboot, Windows will auto-login and the VDD virtual display
   will be active — no physical monitor or manual login needed.

5. Verify from WSL2:

   ```bash
   ls -la /dev/kfd /dev/dri/render*
   # Should appear immediately with no physical session required
   ```

### Why this works

| Component | Role |
| --- | --- |
| VirtualDrivers VDD | Creates a virtual monitor → WDDM initializes GPU fully → `/dev/kfd` appears in WSL2 |
| Windows auto-login | Ensures an active user session owns the display after every reboot |
| HIP SDK WSL component | Bridges the Windows WDDM GPU into the WSL2 kernel namespace |

### Comparison of approaches

| Method | GPU active without monitor? | Survives reboot? | Extra software? |
| --- | --- | --- | --- |
| Physical console | Requires monitor | Manual login | None |
| Auto-login only (no VDD) | Unreliable — GPU may sleep | Yes | Autologon |
| **VDD (VirtualDrivers) + auto-login** | **Yes** | **Yes** | **VDC app (winget installable)** |
| Full Parsec app | Yes (includes its own VDD) | Yes | Parsec account + app |
