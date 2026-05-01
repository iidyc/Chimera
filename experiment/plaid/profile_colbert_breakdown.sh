#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  experiment/plaid/profile_colbert_breakdown.sh --dataset <name> --pair <ncells,ndocs> [--pair <ncells,ndocs> ...] [options]

Description:
  Run ColBERT on one dataset for one or more explicitly selected configurations
  and emit one aggregated stage/H2D transfer breakdown per configuration.

Options:
  --dataset <name>                Dataset under dataset/. Required.
  --pair <ncells,ndocs>           Repeatable selected configuration.
  --k <top_k>                     Retrieval depth. Default: 100
  --implementation-label <label>  Output label. Default: colbert_profile
  --compressed-embeddings-storage <cpu|gpu>
                                  Default: cpu
  --gpu-index-resident            Requires --compressed-embeddings-storage gpu.
  --profile-breakdown-csv <path>  Output CSV. Default:
                                  profiling/<dataset>/<label>/profile_breakdown.csv
  --env-name <name>               Passed through to bench_colbert.sh.
  --experiment-name <name>        Passed through to bench_colbert.sh.
  --index-name <name>             Passed through to bench_colbert.sh.
  --warmup <count>                Passed through to bench_colbert.sh.
  --dry-run                       Print the generated command without executing it.
  -h, --help                      Show this help message.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
bench_script="${repo_root}/experiment/plaid/bench_colbert.sh"

[[ -x "${bench_script}" ]] || die "missing executable helper script: ${bench_script}"

dataset=""
k=100
implementation_label="colbert_profile"
compressed_embeddings_storage="cpu"
gpu_index_resident=0
profile_breakdown_csv=""
dry_run=0
pass_through_args=()
pairs=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || die "missing value for --dataset"
            dataset="$2"
            shift 2
            ;;
        --pair)
            [[ $# -ge 2 ]] || die "missing value for --pair"
            pairs+=("$2")
            shift 2
            ;;
        --k)
            [[ $# -ge 2 ]] || die "missing value for --k"
            k="$2"
            shift 2
            ;;
        --implementation-label)
            [[ $# -ge 2 ]] || die "missing value for --implementation-label"
            implementation_label="$2"
            shift 2
            ;;
        --compressed-embeddings-storage)
            [[ $# -ge 2 ]] || die "missing value for --compressed-embeddings-storage"
            compressed_embeddings_storage="$2"
            shift 2
            ;;
        --gpu-index-resident)
            gpu_index_resident=1
            shift
            ;;
        --profile-breakdown-csv)
            [[ $# -ge 2 ]] || die "missing value for --profile-breakdown-csv"
            profile_breakdown_csv="$2"
            shift 2
            ;;
        --env-name|--experiment-name|--index-name|--warmup)
            [[ $# -ge 2 ]] || die "missing value for $1"
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

[[ -n "$dataset" ]] || die "--dataset is required"
[[ ${#pairs[@]} -gt 0 ]] || die "at least one --pair is required"

case "$compressed_embeddings_storage" in
    cpu|gpu)
        ;;
    *)
        die "--compressed-embeddings-storage must be one of: cpu, gpu"
        ;;
esac

if [[ $gpu_index_resident -eq 1 && "$compressed_embeddings_storage" != "gpu" ]]; then
    die "--gpu-index-resident requires --compressed-embeddings-storage gpu"
fi

if [[ -z "$profile_breakdown_csv" ]]; then
    profile_breakdown_csv="${repo_root}/profiling/${dataset}/${implementation_label}/profile_breakdown.csv"
fi

tmp_config="$(mktemp "${TMPDIR:-/tmp}/colbert_profile_config.XXXXXX.csv")"
cleanup() {
    rm -f "${tmp_config}"
}
trap cleanup EXIT

{
    printf 'label,ncells,ndocs\n'
    pair_idx=0
    for pair in "${pairs[@]}"; do
        IFS=',' read -r ncells ndocs <<< "${pair}"
        [[ -n "${ncells}" && -n "${ndocs}" ]] || die "invalid --pair value: ${pair}"
        printf 'p%d,%s,%s\n' "${pair_idx}" "${ncells}" "${ndocs}"
        pair_idx=$((pair_idx + 1))
    done
} > "${tmp_config}"

cmd=(
    "${bench_script}"
    --dataset "${dataset}"
    --config-file "${tmp_config}"
    --k "${k}"
    --implementation-label "${implementation_label}"
    --compressed-embeddings-storage "${compressed_embeddings_storage}"
    --profile-breakdown-csv "${profile_breakdown_csv}"
)

if [[ $gpu_index_resident -eq 1 ]]; then
    cmd+=(--gpu-index-resident)
fi

if [[ ${#pass_through_args[@]} -gt 0 ]]; then
    cmd+=("${pass_through_args[@]}")
fi

if [[ $dry_run -eq 1 ]]; then
    cmd+=(--dry-run)
fi

echo "[profile] command=$(printf '%q ' "${cmd[@]}")"
"${cmd[@]}"
