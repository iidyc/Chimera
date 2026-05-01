#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  experiment/chimera/bench_gpu_search.sh [--dataset <name> ...] [options]
  experiment/chimera/bench_gpu_search.sh [dataset_name ...] [options]

Description:
  Benchmark a Chimera gpu_search binary on one or more datasets and write
  raw per-run logs only.

  By default, the script benchmarks:
    hotpot
    lotte
    msmarco
  It reads search configurations from profiling/gpu_search_config.csv and writes
  a unique log file for each dataset run:
    log/bench/<implementation>/<dataset>/benchmark_<run_id>.log

  The log records the intended structured-output paths, but this script does
  not generate CSV tables itself. Use experiment/common/parse_bench_logs.py
  later to regenerate benchmark_results_*.csv and pareto_frontier_*.csv from
  the logs.

Options:
  --dataset <name>         Dataset under dataset/. Repeatable.
  --version <gpu_search|gpu_search_nosum|gpu_search_nolut|gpu_search_nolut_nosum>
                           Chimera search binary. Default: gpu_search
  --implementation-label <label>
                           Output folder label. Default: selected --version value.
  --config-file <path>     Config CSV. Default: profiling/gpu_search_config.csv
  --build-dir <path>       Dedicated build directory. Default: Chimera/build
  --binary <path>          Search binary. Overrides --build-dir and --version.
                           Default: selected gpu_search binary under <build-dir>.
  --output-dir <path>      Structured-output root recorded in log metadata for
                           experiment/common/parse_bench_logs.py. Default: profiling
  --log-dir <path>         Log output directory. Default: log/bench
  --k <top_k>              Final retrieval depth / recall depth. Default: 100
  --nq <count>             Evaluation queries timed after warmup; evaluation restarts from query 0. Default: -1
  --warmup <count>         Warmup query count before timed evaluation restarts from query 0. Default: 5
  --concurrent-queries <count>
                           Pass --concurrent-queries to implementations that support query slots.
  --stage3-threads <count> Pass --stage3-threads to implementations that support it.
  --dry-run                Print planned commands without executing them.
  -h, --help               Show this help message.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

log_line() {
    local line="$1"
    if [[ $dry_run -eq 1 ]]; then
        echo "$line"
    else
        echo "$line" | tee -a "$current_log_file"
    fi
}

make_run_id() {
    local timestamp
    local suffix
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

    suffix="pid$$"

    printf '%s_%s' "$timestamp" "$suffix"
}

require_positive_int() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer"
}

validate_dataset_name() {
    local dataset_name="$1"
    case "$dataset_name" in
        ""|.|..|*/*)
            die "dataset name must be a simple directory name under dataset/: ${dataset_name}"
            ;;
    esac
}

validate_path_component() {
    local value="$1"
    local name="$2"
    case "$value" in
        ""|.|..|*/*)
            die "${name} must be a simple path component: ${value}"
            ;;
    esac
}

get_cpu_model() {
    local cpu_model=""
    if command -v lscpu >/dev/null 2>&1; then
        cpu_model="$(lscpu | awk -F: '/Model name:/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
    fi
    if [[ -z "$cpu_model" && -r /proc/cpuinfo ]]; then
        cpu_model="$(awk -F: '/model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
    fi
    if [[ -z "$cpu_model" ]]; then
        cpu_model="unknown"
    fi
    echo "$cpu_model"
}

get_gpu_count_used() {
    local cuda_visible_devices="${CUDA_VISIBLE_DEVICES-}"
    if [[ -n "$cuda_visible_devices" ]]; then
        local cleaned="${cuda_visible_devices// /}"
        local count=0
        local gpu_id=""
        IFS=',' read -r -a gpu_ids <<< "$cleaned"
        for gpu_id in "${gpu_ids[@]}"; do
            [[ -n "$gpu_id" ]] && ((count += 1))
        done
        echo "$count"
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_count_output=""
        if gpu_count_output="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)"; then
            printf '%s\n' "$gpu_count_output" | sed '/^$/d' | wc -l | tr -d '[:space:]'
            return
        fi
    fi

    echo "unknown"
}

print_hardware_info() {
    log_line "[driver] cpu_model=$(get_cpu_model)"
    log_line "[driver] cpu_threads_available=$(nproc)"
    log_line "[driver] omp_num_threads=${OMP_NUM_THREADS-<unset>}"
    log_line "[driver] cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
    log_line "[driver] gpu_count_used=$(get_gpu_count_used)"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_line "[driver] nvidia_smi=unavailable"
        return
    fi

    local gpu_query_output=""
    local gpu_info_lines=()
    local gpu_line=""
    if ! gpu_query_output="$(nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader 2>/dev/null)"; then
        log_line "[driver] nvidia_smi_query=failed"
        return
    fi

    mapfile -t gpu_info_lines <<< "$gpu_query_output"
    log_line "[driver] host_gpu_count=${#gpu_info_lines[@]}"
    for gpu_line in "${gpu_info_lines[@]}"; do
        [[ -n "$gpu_line" ]] || continue
        log_line "[driver] host_gpu=${gpu_line}"
    done
}

count_gpu_search_configs() {
    local config_path="$1"
    awk -F, '
        {
            gsub(/\r/, "", $0)
            if ($0 ~ /^[[:space:]]*$/) {
                next
            }
            label = $1
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", label)
            if (label == "" || label == "label" || substr(label, 1, 1) == "#") {
                next
            }
            count++
        }
        END {
            print count + 0
        }
    ' "$config_path"
}

default_binary_for_version() {
    local version_name="$1"
    case "$version_name" in
        gpu_search)
            printf '%s/gpu_search' "$build_dir"
            ;;
        gpu_search_nosum)
            printf '%s/gpu_search_nosum' "$build_dir"
            ;;
        gpu_search_nolut)
            printf '%s/gpu_search_nolut' "$build_dir"
            ;;
        gpu_search_nolut_nosum)
            printf '%s/gpu_search_nolut_nosum' "$build_dir"
            ;;
        *)
            die "unsupported gpu_search binary: ${version_name}"
            ;;
    esac
}

resolve_source_index_dir() {
    local dataset_dir="$1"
    local candidate=""
    local legacy_prefix="gpu_""mvr"
    local candidates=(
        "${dataset_dir}/gpu_search_2m"
        "${dataset_dir}/gpu_search_1m"
        "${dataset_dir}/gpu_search"
        "${dataset_dir}/gpu_search_index"
        "${dataset_dir}/${legacy_prefix}_2m"
        "${dataset_dir}/${legacy_prefix}_1m"
        "${dataset_dir}/${legacy_prefix}"
        "${dataset_dir}/${legacy_prefix}_index"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -d "$candidate" && -f "${candidate}/doclens.bin" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done

    return 1
}

benchmark_dataset() {
    local dataset_name="$1"

    validate_dataset_name "$dataset_name"

    local dataset_dir="${repo_root}/dataset/${dataset_name}"
    local raw_dir="${dataset_dir}/raw"
    local source_index_dir=""
    local query_file="${raw_dir}/query.bin"
    local index_doclens_file="${source_index_dir}/doclens.bin"
    local gt_file="${raw_dir}/gt.tsv"
    local impl_output_dir="${output_dir}/${dataset_name}/${implementation_label}"
    local impl_log_dir="${log_dir}/${implementation_label}/${dataset_name}"
    local run_id=""
    local results_csv=""
    local pareto_csv=""
    local log_file=""
    local config_count=""
    local status=0
    local run_start_utc=""
    local run_end_utc=""
    local command=()

    [[ -d "$dataset_dir" ]] || die "dataset directory not found: ${dataset_dir}"
    [[ -f "$query_file" ]] || die "missing query file: ${query_file}"
    [[ -f "$gt_file" ]] || die "missing ground truth file: ${gt_file}"
    source_index_dir="$(resolve_source_index_dir "$dataset_dir")" || \
        die "missing Chimera index directory under ${dataset_dir}"
    index_doclens_file="${source_index_dir}/doclens.bin"

    [[ -f "$index_doclens_file" ]] || die "missing index doclens file: ${index_doclens_file}"

    run_id="$(make_run_id)"
    results_csv="${impl_output_dir}/benchmark_results_${run_id}.csv"
    pareto_csv="${impl_output_dir}/pareto_frontier_${run_id}.csv"
    log_file="${impl_log_dir}/benchmark_${run_id}.log"

    current_log_file="$log_file"

    if [[ $dry_run -eq 0 ]]; then
        mkdir -p "$impl_log_dir"
        : > "$log_file"
    fi

    log_line "[driver] repo_root=${repo_root}"
    log_line "[driver] implementation=${implementation_label}"
    log_line "[driver] dataset=${dataset_name}"
    log_line "[driver] run_id=${run_id}"
    log_line "[driver] config_file=${config_file}"
    log_line "[driver] build_dir=${build_dir}"
    log_line "[driver] binary=${binary}"
    log_line "[driver] query_file=${query_file}"
    log_line "[driver] index_doclens_file=${index_doclens_file}"
    log_line "[driver] gt_file=${gt_file}"
    log_line "[driver] source_index_dir=${source_index_dir}"
    log_line "[driver] output_dir=${output_dir}"
    log_line "[driver] results_csv=${results_csv}"
    log_line "[driver] pareto_csv=${pareto_csv}"
    log_line "[driver] log_file=${log_file}"
    log_line "[driver] parser_script=${repo_root}/experiment/common/parse_bench_logs.py"
    log_line "[driver] k=${k}"
    log_line "[driver] nq=${nq}"
    log_line "[driver] warmup=${warmup}"
    print_hardware_info
    config_count="$(count_gpu_search_configs "$config_file")"
    log_line "[driver] config_count=${config_count}"

    command=(
        "$binary"
        --query "$query_file"
        --gt "$gt_file"
        --index "$source_index_dir"
        --k "$k"
        --nq "$nq"
        --warmup "$warmup"
        --config-file "$config_file"
    )
    if [[ -n "$concurrent_queries" ]]; then
        command+=(--concurrent-queries "$concurrent_queries")
    fi
    if [[ -n "$stage3_threads" ]]; then
        command+=(--stage3-threads "$stage3_threads")
    fi

    log_line "[run] dataset=${dataset_name} batched_configs=${config_count}"
    log_line "[run] command=$(printf '%q ' "${command[@]}")"

    if [[ $dry_run -eq 1 ]]; then
        return
    fi

    run_start_utc="$(date -u +%Y%m%dT%H%M%SZ)"
    log_line "[run] start_utc=${run_start_utc}"
    log_line "[run] live_output=enabled"

    set +e
    if command -v stdbuf >/dev/null 2>&1; then
        stdbuf -oL -eL "${command[@]}" 2>&1 | tee -a "$current_log_file"
        status=${PIPESTATUS[0]}
    else
        "${command[@]}" 2>&1 | tee -a "$current_log_file"
        status=${PIPESTATUS[0]}
    fi
    set -e

    run_end_utc="$(date -u +%Y%m%dT%H%M%SZ)"
    log_line "[run] end_utc=${run_end_utc}"
    log_line "[run] exit_code=${status}"

    if [[ $status -ne 0 ]]; then
        die "gpu_search_${version} failed for dataset=${dataset_name}; see ${current_log_file}"
    fi

    log_line "[summary] raw_log=${log_file}"
    log_line "[summary] parse_with=$(printf '%q ' "${repo_root}/experiment/common/parse_bench_logs.py" "$log_file")"
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

config_file="${repo_root}/profiling/gpu_search_config.csv"
build_dir="${repo_root}/Chimera/build"
binary=""
output_dir="${repo_root}/profiling"
log_dir="${repo_root}/log/bench"
version="gpu_search"
implementation_label=""
k=100
nq=-1
warmup=5
concurrent_queries=""
stage3_threads=""
dry_run=0
dataset_set=0
binary_set=0
implementation_label_set=0
current_log_file=""
datasets=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || die "missing value for --dataset"
            datasets+=("$2")
            dataset_set=1
            shift 2
            ;;
        --version)
            [[ $# -ge 2 ]] || die "missing value for --version"
            version="$2"
            shift 2
            ;;
        --implementation-label)
            [[ $# -ge 2 ]] || die "missing value for --implementation-label"
            implementation_label="$2"
            implementation_label_set=1
            shift 2
            ;;
        --config-file)
            [[ $# -ge 2 ]] || die "missing value for --config-file"
            config_file="$2"
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || die "missing value for --build-dir"
            build_dir="$2"
            shift 2
            ;;
        --binary)
            [[ $# -ge 2 ]] || die "missing value for --binary"
            binary="$2"
            binary_set=1
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || die "missing value for --output-dir"
            output_dir="$2"
            shift 2
            ;;
        --log-dir)
            [[ $# -ge 2 ]] || die "missing value for --log-dir"
            log_dir="$2"
            shift 2
            ;;
        --k)
            [[ $# -ge 2 ]] || die "missing value for --k"
            k="$2"
            shift 2
            ;;
        --nq)
            [[ $# -ge 2 ]] || die "missing value for --nq"
            nq="$2"
            shift 2
            ;;
        --warmup)
            [[ $# -ge 2 ]] || die "missing value for --warmup"
            warmup="$2"
            shift 2
            ;;
        --concurrent-queries)
            [[ $# -ge 2 ]] || die "missing value for --concurrent-queries"
            concurrent_queries="$2"
            shift 2
            ;;
        --stage3-threads)
            [[ $# -ge 2 ]] || die "missing value for --stage3-threads"
            stage3_threads="$2"
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
        -*)
            die "unknown option: $1"
            ;;
        *)
            datasets+=("$1")
            dataset_set=1
            shift
            ;;
    esac
done

if [[ $dataset_set -eq 0 ]]; then
    datasets=(hotpot lotte msmarco)
fi

valid_versions="gpu_search, gpu_search_nosum, gpu_search_nolut, gpu_search_nolut_nosum"
case "$version" in
    gpu_search|gpu_search_nosum|gpu_search_nolut|gpu_search_nolut_nosum)
        ;;
    *)
        die "--version must be one of: ${valid_versions}"
        ;;
esac

if [[ $implementation_label_set -eq 0 ]]; then
    case "$version" in
        gpu_search) implementation_label="chimera" ;;
        gpu_search_nosum) implementation_label="chimera_nosum" ;;
        gpu_search_nolut) implementation_label="chimera_nolut" ;;
        gpu_search_nolut_nosum) implementation_label="chimera_nolut_nosum" ;;
    esac
fi

validate_path_component "$implementation_label" "implementation label"

if [[ $binary_set -eq 0 ]]; then
    binary="$(default_binary_for_version "$version")"
fi

require_positive_int "--k" "$k"
[[ "$nq" =~ ^-?[0-9]+$ ]] || die "--nq must be an integer"
require_positive_int "--warmup" "$warmup"
if [[ -n "$concurrent_queries" ]]; then
    require_positive_int "--concurrent-queries" "$concurrent_queries"
    (( concurrent_queries > 0 )) || die "--concurrent-queries must be > 0"
fi
if [[ -n "$stage3_threads" ]]; then
    require_positive_int "--stage3-threads" "$stage3_threads"
fi
(( k > 0 )) || die "--k must be > 0"
(( warmup >= 0 )) || die "--warmup must be >= 0"
(( nq >= -1 )) || die "--nq must be >= -1"

[[ -f "$config_file" ]] || die "config CSV not found: ${config_file}"
if [[ $dry_run -eq 0 ]]; then
    [[ -x "$binary" ]] || die "search binary not found or not executable: ${binary}. Build the selected Chimera target in a dedicated build directory such as ${build_dir}"
elif [[ ! -x "$binary" ]]; then
    echo "[dry-run] warning: binary not found or not executable: ${binary}"
fi

for dataset_name in "${datasets[@]}"; do
    benchmark_dataset "$dataset_name"
done

if [[ $dry_run -eq 1 ]]; then
    echo
    echo "Dry run complete."
else
    echo
    echo "Benchmarking finished."
fi
