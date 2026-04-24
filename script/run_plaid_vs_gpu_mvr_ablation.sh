#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  script/run_plaid_vs_gpu_mvr_ablation.sh [options]

Description:
  Prepare ablation experiments for GPU-MVR.

Modes:
  kernel
    Compare kernel-optimized vs baseline variants:
      - v8 vs v3
      - v6 vs v4

  overlap
    Prepare the overlap-vs-non-overlap experiment plan.
    Source/build-system support is still missing.

The script defaults to dry-run mode. Pass --execute to run the supported
kernel-ablation commands.

Options:
  --mode <kernel|overlap>  Default: kernel
  --dataset <name>         Dataset. Repeatable. Default: lotte hotpot msmarco
  --k <top_k>              Repeatable. Default: 10 and 100
  --gpu-config <path>      GPU-MVR config CSV. Default: profiling/gpu_mvr_config.csv
  --build-dir <path>       GPU-MVR build dir. Default: gpu-mvr/build
  --output-root <path>     Output profiling root.
                           Default: profiling/experiments/plaid_vs_gpu_mvr
  --log-root <path>        Output log root.
                           Default: log/bench/experiments/plaid_vs_gpu_mvr
  --warmup <count>         Warmup queries. Default: 5
  --nq <count>             Timed queries. Default: -1
  --execute                Execute supported commands
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

bench_gpu="${script_dir}/bench_gpu_mvr.sh"
parse_logs="${script_dir}/parse_bench_logs.py"

[[ -x "$bench_gpu" ]] || die "missing executable helper script: $bench_gpu"
[[ -f "$parse_logs" ]] || die "missing parser: $parse_logs"

mode="kernel"
datasets=(lotte hotpot msmarco)
k_values=(10 100)
gpu_config="${repo_root}/profiling/gpu_mvr_config.csv"
build_dir="${repo_root}/gpu-mvr/build"
output_root="${repo_root}/profiling/experiments/plaid_vs_gpu_mvr"
log_root="${repo_root}/log/bench/experiments/plaid_vs_gpu_mvr"
warmup=5
nq=-1
execute=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            [[ $# -ge 2 ]] || die "missing value for --mode"
            mode="$2"
            shift 2
            ;;
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

case "$mode" in
    kernel)
        for k in "${k_values[@]}"; do
            profile_root="${output_root}/ablation_kernel/k${k}"
            profile_log_root="${log_root}/ablation_kernel/k${k}"
            for version in v3 v4 v6 v8; do
                cmd=(
                    "$bench_gpu"
                    --version "$version"
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
            done
            run_cmd python "$parse_logs" --mode gpu-mvr \
                "${profile_log_root}/gpu_search_v3" \
                "${profile_log_root}/gpu_search_v4" \
                "${profile_log_root}/gpu_search_v6" \
                "${profile_log_root}/gpu_search_v8"
        done
        ;;
    overlap)
        cat <<EOF
[todo] Overlap vs non-overlap ablation is blocked on build-system support.
[todo] Current state:
       GPU_MVR_OVERLAP_STAGE23 is enabled globally in gpu-mvr/CMakeLists.txt.
[todo] Needed addition:
       a CMake option or explicit overlap/non-overlap targets for v6 and v8.
[todo] Intended future build knobs:
       -DGPU_MVR_ENABLE_STAGE23_OVERLAP=ON|OFF
[todo] Intended future binaries:
       gpu_search_v6_overlap / gpu_search_v6_nooverlap
       gpu_search_v8_overlap / gpu_search_v8_nooverlap
EOF
        ;;
    *)
        die "--mode must be one of: kernel, overlap"
        ;;
esac
