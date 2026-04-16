#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  script/build_all_gpu_mvr_indices.sh [--n-clusters <count>] [--dry-run]

Description:
  Build GPU-MVR indices for the default dataset set:
    lotte
    hotpot
    msmarco

  This is a thin wrapper around script/build_gpu_mvr_index.sh and keeps the
  per-dataset output locations and log files unchanged.

Arguments:
  --n-clusters <count> Number of IVF/CAGRA centroids to pass through.
                       Default: 2097152.
  --dry-run            Print the commands without running them.
  -h, --help           Show this help message.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
single_build_script="${script_dir}/build_gpu_mvr_index.sh"

[[ -x "$single_build_script" ]] || die "missing executable helper script: ${single_build_script}"

n_clusters="2097152"
dry_run=0
datasets=(lotte hotpot msmarco)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-clusters)
            [[ $# -ge 2 ]] || die "missing value for --n-clusters"
            n_clusters="$2"
            shift 2
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

[[ "$n_clusters" =~ ^[0-9]+$ ]] || die "--n-clusters must be a positive integer"
(( n_clusters > 0 )) || die "--n-clusters must be > 0"

for dataset in "${datasets[@]}"; do
    cmd=(
        "$single_build_script"
        --dataset "$dataset"
        --n-clusters "$n_clusters"
    )

    if [[ $dry_run -eq 1 ]]; then
        cmd+=(--dry-run)
    fi

    echo "[batch] dataset=${dataset}"
    echo "[batch] command=$(printf '%q ' "${cmd[@]}")"
    "${cmd[@]}"
done
