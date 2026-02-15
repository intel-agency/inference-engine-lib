#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/agent-self-check.sh [--fix]

Options:
  --fix    Run Trunk auto-fixes first, then re-run Trunk checks.
EOF
}

run_fix=false
if [[ "${1:-}" == "--fix" ]]; then
  run_fix=true
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage >&2
  exit 2
fi

for cmd in trunk dotnet; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found on PATH: $cmd" >&2
    exit 127
  fi
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

if [[ "$run_fix" == true ]]; then
  echo "==> trunk check --all --fix"
  trunk check --all --fix
fi

echo "==> trunk check --all"
trunk check --all

echo "==> dotnet build"
dotnet build

echo "==> dotnet test"
dotnet test

echo "Self-check completed successfully."
