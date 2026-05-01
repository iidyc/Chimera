#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  experiment/plaid/bench_all_colbert.sh [options]

Description:
  Benchmark ColBERT across the default dataset set:
    lotte
    hotpot
    msmarco

  This is a thin wrapper around experiment/plaid/bench_colbert.sh. It preserves the
  default ColBERT output layout from the single-run driver, e.g.:
    profiling/<dataset>/colbert/

Options:
  --dataset <name>           Dataset to benchmark. Repeatable.
                             Accepts hotspot as an alias for hotpot.
  --config-file <path>       Passed through to bench_colbert.sh.
  --env-name <name>          Passed through to bench_colbert.sh.
  --search-script <path>     Passed through to bench_colbert.sh.
  --experiment-name <name>   Passed through to bench_colbert.sh.
  --index-name <name>        Passed through to bench_colbert.sh.
  --implementation-label <label>
                             Passed through to bench_colbert.sh.
  --output-dir <path>        Passed through to bench_colbert.sh.
  --log-dir <path>           Passed through to bench_colbert.sh.
  --k <top_k>                Passed through to bench_colbert.sh.
  --warmup <count>           Passed through to bench_colbert.sh.
  --compressed-embeddings-storage <cpu|gpu>
                             Passed through to bench_colbert.sh.
  --gpu-index-resident       Passed through to bench_colbert.sh.
  --profile-breakdown-csv <path>
                             Passed through to bench_colbert.sh.
  --dry-run                  Print commands without executing them.
  -h, --help                 Show this help message.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

normalize_dataset_name() {
    local dataset_name="$1"
    case "$dataset_name" in
        hotspot)
            printf '%s' "hotpot"
            ;;
        *)
            printf '%s' "$dataset_name"
            ;;
    esac
}

append_unique() {
    local value="$1"
    shift

    local existing=""
    for existing in "$@"; do
        if [[ "$existing" == "$value" ]]; then
            return 0
        fi
    done

    return 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
single_bench_script="${script_dir}/bench_colbert.sh"

[[ -x "$single_bench_script" ]] || die "missing executable helper script: ${single_bench_script}"

datasets=(lotte hotpot msmarco)
dataset_set=0
dry_run=0
pass_through_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || die "missing value for --dataset"
            normalized_dataset="$(normalize_dataset_name "$2")"
            if [[ $dataset_set -eq 0 ]]; then
                datasets=()
                dataset_set=1
            fi
            if ! append_unique "$normalized_dataset" "${datasets[@]}"; then
                datasets+=("$normalized_dataset")
            fi
            shift 2
            ;;
        --config-file|--env-name|--search-script|--experiment-name|--index-name|--implementation-label|--output-dir|--log-dir|--k|--warmup|--compressed-embeddings-storage|--profile-breakdown-csv)
            [[ $# -ge 2 ]] || die "missing value for $1"
            pass_through_args+=("$1" "$2")
            shift 2
            ;;
        --gpu-index-resident)
            pass_through_args+=("$1")
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

[[ ${#datasets[@]} -gt 0 ]] || die "no datasets selected"

xdg_cache_home="${XDG_CACHE_HOME:-/tmp/${USER:-user}/colbert_mamba_cache}"
cmd=("$single_bench_script")

for dataset in "${datasets[@]}"; do
    cmd+=(--dataset "$dataset")
done

if [[ ${#pass_through_args[@]} -gt 0 ]]; then
    cmd+=("${pass_through_args[@]}")
fi

if [[ $dry_run -eq 1 ]]; then
    cmd+=(--dry-run)
fi

cmd=(env "XDG_CACHE_HOME=${xdg_cache_home}" "${cmd[@]}")

echo "[batch] datasets=${datasets[*]}"
echo "[batch] xdg_cache_home=${xdg_cache_home}"
echo "[batch] command=$(printf '%q ' "${cmd[@]}")"
"${cmd[@]}"
