#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${repo_root}/build"

command -v conda >/dev/null 2>&1 || {
    echo "error: conda is not available on PATH" >&2
    exit 1
}

env_prefix="$(conda env list | awk '$1 == "chimera" { print $NF; exit }')"
if [[ -z "${env_prefix}" ]]; then
    echo "error: conda environment 'chimera' does not exist" >&2
    exit 1
fi

conda run -p "${env_prefix}" cmake \
    -S "${repo_root}" \
    -B "${build_dir}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release

conda run -p "${env_prefix}" cmake \
    --build "${build_dir}" \
    -j "$(nproc)"
