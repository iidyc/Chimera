#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  script/run_plaid_large_vram.sh [options]

Description:
  Prepare the PLAID large-VRAM experiment by comparing:
    - plaid_cpu_hosted
    - plaid_gpu_resident

  The script defaults to dry-run mode. Pass --execute to actually run it.

Options:
  --dataset <name>         Dataset. Repeatable. Default: lotte hotpot msmarco
  --k <top_k>              Repeatable. Default: 10 and 100
  --config-file <path>     ColBERT config CSV. Default: profiling/colbert_config.csv
  --output-root <path>     Output profiling root.
                           Default: profiling/experiments/plaid_vs_gpu_mvr
  --log-root <path>        Output log root.
                           Default: log/bench/experiments/plaid_vs_gpu_mvr
  --warmup <count>         Warmup queries. Default: 5
  --execute                Execute commands instead of printing them
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

bench_colbert="${script_dir}/bench_all_colbert.sh"
bench_colbert_gpu="${script_dir}/bench_all_colbert_gpu_resident.sh"
compare_script="${script_dir}/compare_plaid_vs_gpu_mvr.py"

[[ -x "$bench_colbert" ]] || die "missing executable helper script: $bench_colbert"
[[ -x "$bench_colbert_gpu" ]] || die "missing executable helper script: $bench_colbert_gpu"
[[ -f "$compare_script" ]] || die "missing compare script: $compare_script"

datasets=(lotte hotpot msmarco)
k_values=(10 100)
config_file="${repo_root}/profiling/colbert_config.csv"
output_root="${repo_root}/profiling/experiments/plaid_vs_gpu_mvr"
log_root="${repo_root}/log/bench/experiments/plaid_vs_gpu_mvr"
warmup=5
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
        --config-file)
            [[ $# -ge 2 ]] || die "missing value for --config-file"
            config_file="$2"
            shift 2
            ;;
        --output-root)
            [[ $# -ge 2 ]] || die "missing value for --output-root"
            output_root="$2"
            shift 2
            ;;
        --log-root)
            [[ $# -ge 2 ]] || die "missing value for --log-root"
            log_root="$2"
            shift 2
            ;;
        --warmup)
            [[ $# -ge 2 ]] || die "missing value for --warmup"
            warmup="$2"
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

for k in "${k_values[@]}"; do
    profile_root="${output_root}/plaid_large_vram/k${k}"
    profile_log_root="${log_root}/plaid_large_vram/k${k}"

    cmd=(
        "$bench_colbert"
        --config-file "$config_file"
        --implementation-label plaid_cpu_hosted
        --output-dir "$profile_root"
        --log-dir "$profile_log_root"
        --k "$k"
        --warmup "$warmup"
    )
    for dataset in "${datasets[@]}"; do
        cmd+=(--dataset "$dataset")
    done
    run_cmd "${cmd[@]}"

    cmd=(
        "$bench_colbert_gpu"
        --config-file "$config_file"
        --implementation-label plaid_gpu_resident
        --output-dir "$profile_root"
        --log-dir "$profile_log_root"
        --k "$k"
        --warmup "$warmup"
    )
    for dataset in "${datasets[@]}"; do
        cmd+=(--dataset "$dataset")
    done
    run_cmd "${cmd[@]}"

    cmd=(
        python "$compare_script"
        --profiling-root "$profile_root"
        --colbert-impl plaid_cpu_hosted
        --colbert-impl plaid_gpu_resident
    )
    for dataset in "${datasets[@]}"; do
        cmd+=(--dataset "$dataset")
    done
    run_cmd "${cmd[@]}"
done
