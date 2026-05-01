#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  experiment/plaid_plus/run_plaid_vs_gpu_search_main.sh [options]

Description:
  Prepare the main PLAID vs. gpu_search benchmark runs for:
    - gpu_search
    - PLAID / ColBERT cpu-hosted
    - PLAID / ColBERT gpu-resident

  The script defaults to dry-run mode and only prints the commands. Pass
  --execute to actually run them later.

Options:
  --dataset <name>         Dataset. Repeatable. Default: lotte hotpot msmarco
  --k <top_k>              Repeatable. Default: 10 and 100
  --gpu-config <path>      gpu_search config CSV. Default: profiling/gpu_search_config.csv
  --colbert-config <path>  ColBERT config CSV. Default: profiling/colbert_config.csv
  --build-dir <path>       gpu_search build dir. Default: Chimera/build
  --output-root <path>     Output profiling root.
                           Default: profiling/experiments/plaid_vs_gpu_search
  --log-root <path>        Output log root.
                           Default: log/bench/experiments/plaid_vs_gpu_search
  --warmup <count>         Warmup queries. Default: 5
  --nq <count>             Timed queries. Default: -1
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
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

bench_gpu="${repo_root}/experiment/chimera/bench_gpu_search.sh"
bench_colbert="${repo_root}/experiment/plaid/bench_all_colbert.sh"
bench_colbert_gpu="${repo_root}/experiment/plaid_plus/bench_all_colbert_gpu_resident.sh"
parse_logs="${repo_root}/experiment/common/parse_bench_logs.py"
compare_script="${repo_root}/experiment/plaid_plus/compare_plaid_vs_gpu_search.py"

[[ -x "$bench_gpu" ]] || die "missing executable helper script: $bench_gpu"
[[ -x "$bench_colbert" ]] || die "missing executable helper script: $bench_colbert"
[[ -x "$bench_colbert_gpu" ]] || die "missing executable helper script: $bench_colbert_gpu"
[[ -f "$parse_logs" ]] || die "missing parser: $parse_logs"
[[ -f "$compare_script" ]] || die "missing compare script: $compare_script"

datasets=(lotte hotpot msmarco)
k_values=(10 100)
gpu_config="${repo_root}/profiling/gpu_search_config.csv"
colbert_config="${repo_root}/profiling/colbert_config.csv"
build_dir="${repo_root}/Chimera/build"
output_root="${repo_root}/profiling/experiments/plaid_vs_gpu_search"
log_root="${repo_root}/log/bench/experiments/plaid_vs_gpu_search"
warmup=5
nq=-1
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
        --gpu-config)
            [[ $# -ge 2 ]] || die "missing value for --gpu-config"
            gpu_config="$2"
            shift 2
            ;;
        --colbert-config)
            [[ $# -ge 2 ]] || die "missing value for --colbert-config"
            colbert_config="$2"
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || die "missing value for --build-dir"
            build_dir="$2"
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
        --nq)
            [[ $# -ge 2 ]] || die "missing value for --nq"
            nq="$2"
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

[[ -f "$gpu_config" ]] || die "missing GPU config CSV: $gpu_config"
[[ -f "$colbert_config" ]] || die "missing ColBERT config CSV: $colbert_config"
[[ ${#datasets[@]} -gt 0 ]] || die "no datasets selected"
[[ ${#k_values[@]} -gt 0 ]] || die "no k values selected"

for k in "${k_values[@]}"; do
    profile_root="${output_root}/main/k${k}"
    profile_log_root="${log_root}/main/k${k}"

    cmd=(
        "$bench_gpu"
        --version gpu_search
        --build-dir "$build_dir"
        --config-file "$gpu_config"
        --output-dir "$profile_root"
        --log-dir "$profile_log_root"
        --k "$k"
        --nq "$nq"
        --warmup "$warmup"
    )
    for dataset in "${datasets[@]}"; do
        cmd+=(--dataset "$dataset")
    done
    run_cmd "${cmd[@]}"

    cmd=(
        "$bench_colbert"
        --config-file "$colbert_config"
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
        --config-file "$colbert_config"
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

    run_cmd python "$parse_logs" --mode chimera \
        "${profile_log_root}/chimera"

    compare_cmd=(
        python "$compare_script"
        --profiling-root "$profile_root"
    )
    for dataset in "${datasets[@]}"; do
        compare_cmd+=(--dataset "$dataset")
    done
    run_cmd "${compare_cmd[@]}"
done
