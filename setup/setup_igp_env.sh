#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  setup/setup_igp_env.sh [--env-name <name>] [--python <version>] [--verify] [--dry-run]

Description:
  Create a Conda environment for the local IGP checkout under ./IGP.
  This installs the CUDA/cuVS toolchain, build dependencies, and the Python
  packages used by the IGP build/evaluation scripts.

Options:
  --env-name <name>   Conda environment name. Default: IGP
  --python <version>  Python version for the new env. Default: 3.11
  --verify            Run post-install checks inside the target env.
  --dry-run           Print commands without executing them.
  -h, --help          Show this help message.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

run_cmd() {
    echo "+ $(printf '%q ' "$@")"
    if [[ $dry_run -eq 0 ]]; then
        "$@"
    fi
}

resolve_env_prefix() {
    local env_prefix
    env_prefix="$(conda env list 2>/dev/null | awk -v env="${env_name}" '
        $1 == env || $2 == env { print $NF }
    ' | tail -n 1)"
    [[ -n "$env_prefix" ]] || die "failed to resolve prefix for env ${env_name}"
    echo "$env_prefix"
}

resolve_env_binary() {
    local env_prefix="$1"
    local binary_name="$2"
    local binary_path="${env_prefix}/bin/${binary_name}"

    if [[ $dry_run -eq 0 && ! -x "$binary_path" ]]; then
        die "missing executable ${binary_path}"
    fi

    echo "$binary_path"
}

run_in_env() {
    local env_prefix="$1"
    local binary_name="$2"
    shift 2

    local binary_path
    binary_path="$(resolve_env_binary "$env_prefix" "$binary_name")"

    run_cmd env \
        "PATH=${env_prefix}/bin:${PATH}" \
        "CONDA_PREFIX=${env_prefix}" \
        "$binary_path" \
        "$@"
}

env_name="IGP"
python_version="3.11"
verify=0
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            [[ $# -ge 2 ]] || die "missing value for --env-name"
            env_name="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || die "missing value for --python"
            python_version="$2"
            shift 2
            ;;
        --verify)
            verify=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

if [[ $dry_run -eq 0 ]]; then
    command -v conda >/dev/null 2>&1 || die "conda is not available on PATH"
fi

run_cmd conda create -n "$env_name" -y "python=${python_version}"
run_cmd conda install -n "$env_name" -y -c conda-forge \
    cxx-compiler cmake ninja make \
    pybind11 spdlog tbb openblas eigen \
    numpy pandas scipy tqdm pip
run_cmd conda install -n "$env_name" -y -c rapidsai -c conda-forge "libcuvs=26.04" "cuda-version=12.9"

if [[ $dry_run -eq 1 ]]; then
    env_prefix="<resolved-env-prefix>"
else
    env_prefix="$(resolve_env_prefix)"
fi

run_in_env "$env_prefix" python -m pip install --upgrade pip
run_in_env "$env_prefix" python -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch
run_in_env "$env_prefix" python -m pip install faiss-gpu-cu12

if [[ $verify -eq 1 ]]; then
    run_in_env "$env_prefix" python --version
    run_in_env "$env_prefix" cmake --version
    run_in_env "$env_prefix" nvcc --version
    run_in_env "$env_prefix" python - <<'PY'
import importlib

mods = ["numpy", "pandas", "tqdm", "faiss", "torch"]
for name in mods:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "ok"))

import torch
print("torch.cuda.is_available", torch.cuda.is_available())
PY
fi

echo
echo "Environment setup script finished for ${env_name}."
echo "To use it:"
echo "  conda activate ${env_name}"
