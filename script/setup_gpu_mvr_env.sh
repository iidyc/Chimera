#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  script/setup_gpu_mvr_env.sh [--env-name <name>] [--python <version>] [--verify] [--dry-run]

Description:
  Create a Conda environment for gpu-mvr that matches the repo README.

Options:
  --env-name <name>   Conda environment name. Default: gpu-mvr
  --python <version>  Python version for the new env. Default: 3.11
  --verify            Run a few post-install checks with conda run.
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

env_name="gpu-mvr"
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
run_cmd conda install -n "$env_name" -y -c rapidsai -c conda-forge "libcuvs=26.04" "cuda-version=12.9"
run_cmd conda install -n "$env_name" -y -c conda-forge cxx-compiler cmake ninja

if [[ $verify -eq 1 ]]; then
    run_cmd conda run -n "$env_name" python --version
    run_cmd conda run -n "$env_name" cmake --version
    run_cmd conda run -n "$env_name" nvcc --version
fi

echo
echo "Environment setup script finished for ${env_name}."
echo "To use it:"
echo "  conda activate ${env_name}"
