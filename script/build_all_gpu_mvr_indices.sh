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

  This is a thin wrapper around script/build_gpu_mvr_index.sh and writes the
  default per-dataset index output under dataset/<name>/gpu_mvr_2m/ plus a
  unique raw build log under log/build/<dataset>/<index_dir>/.

Arguments:
  --n-clusters <count> Number of IVF/CAGRA centroids to pass through.
                       Accepts 2M=2000000 and 1M=1000000.
                       Default: 2000000.
  --log-dir <path>     Passed through to build_gpu_mvr_index.sh.
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

normalize_n_clusters() {
    local value="$1"
    local upper_value="${value^^}"
    case "$upper_value" in
        2M)
            printf '%s' "2000000"
            ;;
        1M)
            printf '%s' "1000000"
            ;;
        *)
            printf '%s' "$value"
            ;;
    esac
}

n_clusters="2000000"
dry_run=0
datasets=(lotte hotpot msmarco)
pass_through_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-clusters)
            [[ $# -ge 2 ]] || die "missing value for --n-clusters"
            n_clusters="$2"
            shift 2
            ;;
        --log-dir)
            [[ $# -ge 2 ]] || die "missing value for --log-dir"
            pass_through_args+=("$1" "$2")
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

n_clusters="$(normalize_n_clusters "$n_clusters")"
[[ "$n_clusters" =~ ^[0-9]+$ ]] || die "--n-clusters must be a positive integer"
(( n_clusters > 0 )) || die "--n-clusters must be > 0"

for dataset in "${datasets[@]}"; do
    cmd=(
        "$single_build_script"
        --dataset "$dataset"
        --n-clusters "$n_clusters"
    )

    if [[ ${#pass_through_args[@]} -gt 0 ]]; then
        cmd+=("${pass_through_args[@]}")
    fi

    if [[ $dry_run -eq 1 ]]; then
        cmd+=(--dry-run)
    fi

    echo "[batch] dataset=${dataset}"
    echo "[batch] command=$(printf '%q ' "${cmd[@]}")"
    "${cmd[@]}"
done
