#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  script/bench_all_gpu_mvr_versions.sh [options]

Description:
  Benchmark GPU-MVR gpu_search versions v0 through v6 across the default
  dataset set:
    lotte
    hotpot
    msmarco

  This is a thin wrapper around script/bench_gpu_mvr.sh. It preserves the
  per-version log layout from the single-version benchmark driver, e.g.:
    log/bench/gpu_search_v0/<dataset>/benchmark_<run_id>.log
    log/bench/gpu_search_v1/<dataset>/benchmark_<run_id>.log
    ...

  Use script/parse_bench_logs.py later to regenerate structured CSV tables from
  the raw logs.

Options:
  --dataset <name>         Dataset to benchmark. Repeatable.
                           Accepts hotspot as an alias for hotpot.
  --version <v0|v1|v2|v3|v4|v5|v6>  Version to benchmark. Repeatable.
                           Default: v0 v1 v2 v3 v4 v5 v6
  --config-file <path>     Passed through to bench_gpu_mvr.sh.
                           Default: profiling/gpu_mvr_config.csv
  --build-dir <path>       Passed through to bench_gpu_mvr.sh.
  --output-dir <path>      Passed through to bench_gpu_mvr.sh.
  --log-dir <path>         Passed through to bench_gpu_mvr.sh.
  --k <top_k>              Passed through to bench_gpu_mvr.sh.
  --nq <count>             Passed through to bench_gpu_mvr.sh.
  --warmup <count>         Passed through to bench_gpu_mvr.sh.
  --dry-run                Print commands without executing them.
  -h, --help               Show this help message.
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
single_bench_script="${script_dir}/bench_gpu_mvr.sh"

[[ -x "$single_bench_script" ]] || die "missing executable helper script: ${single_bench_script}"

datasets=(lotte hotpot msmarco)
versions=(v0 v1 v2 v3 v4 v5 v6)
dataset_set=0
version_set=0
dry_run=0
pass_through_args=()
config_file_set=0

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
        --version)
            [[ $# -ge 2 ]] || die "missing value for --version"
            case "$2" in
                v0|v1|v2|v3|v4|v5|v6)
                    ;;
                *)
                    die "--version must be one of: v0, v1, v2, v3, v4, v5, v6"
                    ;;
            esac
            if [[ $version_set -eq 0 ]]; then
                versions=()
                version_set=1
            fi
            if ! append_unique "$2" "${versions[@]}"; then
                versions+=("$2")
            fi
            shift 2
            ;;
        --config-file|--build-dir|--output-dir|--log-dir|--k|--nq|--warmup)
            [[ $# -ge 2 ]] || die "missing value for $1"
            pass_through_args+=("$1" "$2")
            if [[ "$1" == "--config-file" ]]; then
                config_file_set=1
            fi
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

[[ ${#datasets[@]} -gt 0 ]] || die "no datasets selected"
[[ ${#versions[@]} -gt 0 ]] || die "no versions selected"

if [[ $config_file_set -eq 0 ]]; then
    pass_through_args+=(--config-file "profiling/gpu_mvr_config.csv")
fi

for version in "${versions[@]}"; do
    cmd=(
        "$single_bench_script"
        --version "$version"
    )

    for dataset in "${datasets[@]}"; do
        cmd+=(--dataset "$dataset")
    done

    if [[ ${#pass_through_args[@]} -gt 0 ]]; then
        cmd+=("${pass_through_args[@]}")
    fi

    if [[ $dry_run -eq 1 ]]; then
        cmd+=(--dry-run)
    fi

    echo "[batch] version=${version}"
    echo "[batch] datasets=${datasets[*]}"
    echo "[batch] command=$(printf '%q ' "${cmd[@]}")"
    "${cmd[@]}"
done
