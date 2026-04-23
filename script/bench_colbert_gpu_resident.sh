#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base_script="${script_dir}/bench_colbert.sh"

[[ -x "$base_script" ]] || {
    echo "error: missing executable helper script: ${base_script}" >&2
    exit 1
}

cmd=(
    "$base_script"
    --implementation-label colbert_gpu_resident
    --compressed-embeddings-storage gpu
    --gpu-index-resident
)

if [[ $# -gt 0 ]]; then
    cmd+=("$@")
fi

echo "[gpu-resident] command=$(printf '%q ' "${cmd[@]}")"
exec "${cmd[@]}"
