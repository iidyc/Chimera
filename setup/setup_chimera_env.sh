#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  setup/setup_chimera_env.sh [--env-name <name>] [--python <version>] [--verify] [--dry-run]

Description:
  Create a Conda environment for Chimera that matches the repo README.

Options:
  --env-name <name>   Conda environment name. Default: chimera
  --python <version>  Python version for the new env. Default: 3.11
  --verify            Run a few post-install checks inside the target env.
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
        "$binary_path" \
        "$@"
}

env_name="chimera"
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
run_cmd conda install -n "$env_name" -y -c conda-forge cxx-compiler cmake ninja matplotlib

if [[ $dry_run -eq 1 ]]; then
    env_prefix="<resolved-env-prefix>"
else
    env_prefix="$(resolve_env_prefix)"
fi

if [[ $verify -eq 1 ]]; then
    run_in_env "$env_prefix" python --version
    run_in_env "$env_prefix" cmake --version
    run_in_env "$env_prefix" nvcc --version
fi

echo
echo "Environment setup script finished for ${env_name}."
echo "To use it:"
echo "  conda activate ${env_name}"
