#!/usr/bin/env pwsh

# required AMD GPU ( Supported AMD GPU (RDNA 2/3, CDNA, or GCN 5th-gen+; many older or low-end consumer GPUs are not on the ROCm support matrix).
lspci | grep -i amd

# Linux kernel with amdgpu driver loaded (kernels 5.x+ ship it)
lsmod | grep amdgpu

# DRI render nodes at /dev/dri/renderD128 (and friends).
& /bin/ls -al /dev/dri/renderD128

# User has access to /dev/kfd and /dev/dri/* — typically requires membership in the video and render groups, or running as root.
# Wrong permissions are the #1 "ROCm installed but doesn't work" cause.
& /bin/ls -al /dev/kfd /dev/dri/*