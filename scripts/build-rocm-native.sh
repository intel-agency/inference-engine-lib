#!/usr/bin/env bash
#
# Build ONNX Runtime v1.24.1 native Linux x64 shared libraries with ROCm support.
#
# Requirements:
#   - Docker installed and running
#   - ~10 GB free disk space
#   - 30-60 minutes build time depending on host resources
#
# Outputs:
#   - build/linux-x64/native/*.so (all shared libraries from ROCm build)
#   - Typically includes: libonnxruntime.so, libonnxruntime_providers_rocm.so,
#     libonnxruntime_providers_shared.so
#
# Usage examples:
#   - Local: ./scripts/build-rocm-native.sh
#   - Local clean build: ./scripts/build-rocm-native.sh --clean
#   - CI: invoke this script from a GitHub Actions job step

set -euo pipefail

ONNX_RUNTIME_VERSION="v1.24.1"
# Pin to a specific ROCm image version to keep builds reproducible.
# Update only after confirming compatibility with ONNX_RUNTIME_VERSION above.
ROCM_DOCKER_IMAGE="rocm/dev-ubuntu-22.04:6.2.4"
HIP_ARCHITECTURES="gfx1030;gfx1031;gfx1032;gfx1100;gfx1101"
OUTPUT_DIR="build/linux-x64/native"
WORK_DIR="build/onnxruntime-rocm-build"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  printf '%s ==> %s\n' "$(timestamp)" "$*"
}

error() {
  printf '%s ERROR: %s\n' "$(timestamp)" "$*" >&2
}

usage() {
  cat <<EOF
Compile ONNX Runtime ${ONNX_RUNTIME_VERSION} with ROCm support in Docker.

Usage:
  ./scripts/build-rocm-native.sh [--clean]

Options:
  --clean       Remove previous build artifacts in ${WORK_DIR} before building.
  -h, --help    Show this help text.

Requirements:
  - Docker Desktop/Engine installed and running
  - Network access to pull ${ROCM_DOCKER_IMAGE} and clone ONNX Runtime
  - ~10 GB free disk space

Expected output:
  - ${OUTPUT_DIR}/*.so (all shared libraries from ROCm build)
  - Typically includes libonnxruntime.so, libonnxruntime_providers_rocm.so,
    and libonnxruntime_providers_shared.so

Estimated build time:
  - 30-60 minutes
EOF
}

on_error() {
  local exit_code="$?"
  error "Build process failed with exit code ${exit_code}."
  error "Troubleshooting hints:"
  error "  - Docker not running: Start Docker Desktop and ensure WSL2 integration is enabled."
  error "  - Build failure: Check Docker logs above for compilation errors."
  error "  - Missing artifacts: Check ${WORK_DIR}/onnxruntime/build/Linux/Release/."
  exit "${exit_code}"
}

file_size_bytes() {
  local file_path="$1"
  if stat -c%s "${file_path}" >/dev/null 2>&1; then
    stat -c%s "${file_path}"
  else
    stat -f%z "${file_path}"
  fi
}

format_size() {
  local bytes="$1"
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=iec-i --suffix=B "${bytes}"
  else
    printf '%s bytes' "${bytes}"
  fi
}

validate_minimum_size() {
  local file_path="$1"
  local min_bytes="$2"
  local file_name="$3"

  local size_bytes
  size_bytes="$(file_size_bytes "${file_path}")"

  if (( size_bytes < min_bytes )); then
    error "${file_name} size below minimum threshold: $(format_size "${size_bytes}") (minimum $(format_size "${min_bytes}"))."
    exit 1
  fi

  log "${file_name} meets minimum size requirement: $(format_size "${size_bytes}")"
}

clean_build=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      clean_build=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      error "Unknown argument: $1"
      usage >&2
      exit 2
      ;;
  esac
done

trap on_error ERR

log "[1/5] Validating Docker prerequisites..."
if ! command -v docker >/dev/null 2>&1; then
  error "Required command not found on PATH: docker"
  error "Start Docker Desktop and ensure WSL2 integration is enabled."
  exit 127
fi

if ! docker info >/dev/null 2>&1; then
  error "Docker daemon is not accessible."
  error "Start Docker Desktop and ensure WSL2 integration is enabled."
  exit 127
fi

log "Docker version: $(docker --version)"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

output_dir_path="${repo_root}/${OUTPUT_DIR}"
work_dir_path="${repo_root}/${WORK_DIR}"
onnxruntime_dir="${work_dir_path}/onnxruntime"
compile_script_path="${work_dir_path}/compile-rocm.sh"

log "[2/5] Preparing build directories..."
if [[ "${clean_build}" == true ]]; then
  log "Cleaning previous build workspace at ${work_dir_path}"
  rm -rf "${work_dir_path}"
fi

mkdir -p "${output_dir_path}"
mkdir -p "${work_dir_path}"

if [[ ! -d "${onnxruntime_dir}/.git" ]]; then
  log "Cloning ONNX Runtime ${ONNX_RUNTIME_VERSION} into ${onnxruntime_dir}"
  git clone --depth 1 --branch "${ONNX_RUNTIME_VERSION}" --recursive \
    https://github.com/microsoft/onnxruntime.git \
    "${onnxruntime_dir}"
else
  log "Using existing ONNX Runtime clone at ${onnxruntime_dir}"
  log "Syncing existing ONNX Runtime clone to ${ONNX_RUNTIME_VERSION}"
  git -C "${onnxruntime_dir}" fetch --tags origin "${ONNX_RUNTIME_VERSION}"
  git -C "${onnxruntime_dir}" checkout "${ONNX_RUNTIME_VERSION}"
  git -C "${onnxruntime_dir}" submodule sync --recursive
  git -C "${onnxruntime_dir}" submodule update --init --recursive
fi

if [[ ! -f "${onnxruntime_dir}/build.sh" ]]; then
  error "ONNX Runtime clone validation failed: missing ${onnxruntime_dir}/build.sh"
  exit 1
fi

log "[3/5] Generating in-container ROCm build script..."
cat >"${compile_script_path}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
  build-essential \\
  cmake \\
  git \\
  python3 \\
  python3-dev \\
  python3-pip \\
  libprotobuf-dev \\
  protobuf-compiler

git config --global --add safe.directory /code/onnxruntime

cd /code/onnxruntime
./build.sh \\
  --config Release \\
  --build_shared_lib \\
  --use_rocm \\
  --rocm_home /opt/rocm \\
  --skip_tests \\
  --skip_submodule_sync \\
  --parallel \\
  --cmake_extra_defines CMAKE_HIP_ARCHITECTURES="${HIP_ARCHITECTURES}"
EOF

chmod +x "${compile_script_path}"

log "[4/5] Running Docker ROCm build (estimated 30-60 minutes)..."
set +e
docker run --rm \
  -v "${work_dir_path}:/code" \
  -w /code \
  "${ROCM_DOCKER_IMAGE}" \
  /code/compile-rocm.sh
docker_exit_code="$?"
set -e

if [[ "${docker_exit_code}" -ne 0 ]]; then
  error "Docker build failed with exit code ${docker_exit_code}."
  error "Check Docker logs above for compilation errors."
  exit 1
fi

log "[5/5] Validating and publishing native artifacts..."
artifact_dir="${onnxruntime_dir}/build/Linux/Release"
core_runtime_lib="${artifact_dir}/libonnxruntime.so"
rocm_provider_lib="${artifact_dir}/libonnxruntime_providers_rocm.so"
shopt -s nullglob
artifact_libs=("${artifact_dir}"/*.so)
shopt -u nullglob

if [[ ! -d "${artifact_dir}" ]]; then
  error "Missing artifact directory: ${artifact_dir}"
  error "Build may have failed silently; check Docker logs above."
  exit 1
fi

if [[ "${#artifact_libs[@]}" -eq 0 ]]; then
  error "No shared libraries found in artifact directory: ${artifact_dir}"
  error "Build may have failed silently; check ${artifact_dir}"
  exit 1
fi

if [[ ! -f "${core_runtime_lib}" ]]; then
  error "Missing expected artifact: ${core_runtime_lib}"
  error "Build may have failed silently; check ${artifact_dir}"
  exit 1
fi

if [[ ! -f "${rocm_provider_lib}" ]]; then
  error "Missing expected artifact: ${rocm_provider_lib}"
  error "Build may have failed silently; check ${artifact_dir}"
  exit 1
fi

validate_minimum_size "${core_runtime_lib}" 10485760 "libonnxruntime.so"
validate_minimum_size "${rocm_provider_lib}" 52428800 "libonnxruntime_providers_rocm.so"

cp -f -p "${artifact_dir}"/*.so "${output_dir_path}/"

log "Copied artifacts to ${output_dir_path}"
for copied_file in "${artifact_libs[@]}"; do
  copied_name="$(basename "${copied_file}")"
  copied_size="$(file_size_bytes "${output_dir_path}/${copied_name}")"
  log "  - ${copied_name}: $(format_size "${copied_size}")"
done
log "ROCm native build completed successfully."
