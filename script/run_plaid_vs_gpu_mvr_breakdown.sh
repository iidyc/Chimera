#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  script/run_plaid_vs_gpu_mvr_breakdown.sh [options]

Description:
  Prepare the stage and transfer breakdown experiments for PLAID / ColBERT and
  GPU-MVR.

  Current support:
    - ColBERT: supported now via aggregated breakdown CSVs
    - GPU-MVR v6/v8: source support is still missing for aggregated per-run
      stage / transfer export; this script prints the intended commands and
      output paths for that future support.

  The script defaults to dry-run mode and only prints commands. Pass --execute
  to run the currently supported ColBERT commands.

Options:
  --dataset <name>         Dataset. Repeatable. Default: lotte hotpot msmarco
  --k <top_k>              Repeatable. Default: 10 and 100
  --pair <ncells,ndocs>    ColBERT pair. Repeatable. Default: all pairs from
                           profiling/colbert_config.csv
  --colbert-config <path>  ColBERT config CSV. Default: profiling/colbert_config.csv
  --output-root <path>     Output profiling root.
                           Default: profiling/experiments/plaid_vs_gpu_mvr
  --execute                Execute supported ColBERT commands
  --dry-run                Force dry-run mode
  -h, --help               Show help
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

normalize_dataset_name() {
    case "$1" in
        hotspot) printf '%s' "hotpot" ;;
        *) printf '%s' "$1" ;;
    esac
}

append_unique() {
    local value="$1"
    shift
    local existing=""
    for existing in "$@"; do
        [[ "$existing" == "$value" ]] && return 0
    done
    return 1
}

run_cmd() {
    printf '[cmd] %q ' "$@"
    printf '\n'
    if [[ $execute -eq 1 ]]; then
        "$@"
    fi
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

profile_colbert="${script_dir}/profile_colbert_breakdown.sh"
[[ -x "$profile_colbert" ]] || die "missing executable helper script: $profile_colbert"

datasets=(lotte hotpot msmarco)
k_values=(10 100)
pairs=()
colbert_config="${repo_root}/profiling/colbert_config.csv"
output_root="${repo_root}/profiling/experiments/plaid_vs_gpu_mvr"
execute=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || die "missing value for --dataset"
            value="$(normalize_dataset_name "$2")"
            if ! append_unique "$value" "${datasets[@]}"; then
                datasets+=("$value")
            fi
            shift 2
            ;;
        --k)
            [[ $# -ge 2 ]] || die "missing value for --k"
            [[ "$2" =~ ^[0-9]+$ ]] || die "--k must be a positive integer"
            if [[ ${#k_values[@]} -eq 2 && "${k_values[0]}" == "10" && "${k_values[1]}" == "100" ]]; then
                k_values=()
            fi
            if ! append_unique "$2" "${k_values[@]}"; then
                k_values+=("$2")
            fi
            shift 2
            ;;
        --pair)
            [[ $# -ge 2 ]] || die "missing value for --pair"
            pairs+=("$2")
            shift 2
            ;;
        --colbert-config)
            [[ $# -ge 2 ]] || die "missing value for --colbert-config"
            colbert_config="$2"
            shift 2
            ;;
        --output-root)
            [[ $# -ge 2 ]] || die "missing value for --output-root"
            output_root="$2"
            shift 2
            ;;
        --execute)
            execute=1
            shift
            ;;
        --dry-run)
            execute=0
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

[[ -f "$colbert_config" ]] || die "missing ColBERT config CSV: $colbert_config"

if [[ ${#pairs[@]} -eq 0 ]]; then
    mapfile -t pairs < <(
        python - "$colbert_config" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        print(f"{row['ncells']},{row['ndocs']}")
PY
    )
fi

for k in "${k_values[@]}"; do
    for dataset in "${datasets[@]}"; do
        cpu_csv="${output_root}/breakdown/k${k}/${dataset}/plaid_cpu_hosted/profile_breakdown.csv"
        gpu_csv="${output_root}/breakdown/k${k}/${dataset}/plaid_gpu_resident/profile_breakdown.csv"

        cmd=(
            "$profile_colbert"
            --dataset "$dataset"
            --k "$k"
            --implementation-label plaid_cpu_hosted
            --profile-breakdown-csv "$cpu_csv"
        )
        for pair in "${pairs[@]}"; do
            cmd+=(--pair "$pair")
        done
        run_cmd "${cmd[@]}"

        cmd=(
            "$profile_colbert"
            --dataset "$dataset"
            --k "$k"
            --implementation-label plaid_gpu_resident
            --compressed-embeddings-storage gpu
            --gpu-index-resident
            --profile-breakdown-csv "$gpu_csv"
        )
        for pair in "${pairs[@]}"; do
            cmd+=(--pair "$pair")
        done
        run_cmd "${cmd[@]}"

        cat <<EOF
[todo] gpu_search_v6 aggregate breakdown is not implemented yet.
[todo] intended future output:
       ${output_root}/breakdown/k${k}/${dataset}/gpu_search_v6/profile_breakdown.csv
[todo] intended future command shape:
       gpu_search_v6 --config-file profiling/gpu_mvr_config.csv --k ${k} --profile-breakdown-csv <path>

[todo] gpu_search_v8 aggregate breakdown is not implemented yet.
[todo] intended future output:
       ${output_root}/breakdown/k${k}/${dataset}/gpu_search_v8/profile_breakdown.csv
[todo] intended future command shape:
       gpu_search_v8 --config-file profiling/gpu_mvr_config.csv --k ${k} --profile-breakdown-csv <path>
EOF
    done
done
