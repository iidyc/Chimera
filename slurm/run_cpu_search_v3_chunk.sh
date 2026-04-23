#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  slurm/run_cpu_search_v3_chunk.sh --dataset <name> --config-file <path> --batch-id <id> [options]
  slurm/run_cpu_search_v3_chunk.sh <dataset> --config-file <path> --batch-id <id> [options]

Description:
  Benchmark cpu_search_v3 for one dataset and one config chunk, writing one raw
  benchmark log for that job under slurm/logs.

Options:
  --dataset <name>             Dataset under dataset/.
  --config-file <path>         CSV containing one or more runtime configs.
  --batch-id <id>              Shared batch identifier across all chunk jobs.
  --batch-total-configs <n>    Total config count in the original CSV.
  --binary <path>              Search binary. Default: gpu-mvr/build-cpu-only/cpu_search_v3
  --output-dir <path>          Combined CSV output root. Default: profiling
  --log-dir <path>             Raw log root. Default: slurm/logs
  --implementation-label <id>  Output folder label. Default: cpu_search_v3
  --k <top_k>                  Final retrieval depth / recall depth. Default: 100
  --nq <count>                 Evaluation queries timed after warmup. Default: -1
  --warmup <count>             Warmup queries before timed evaluation. Default: 5
  --omp-threads <count>        OMP_NUM_THREADS value. Default: 36
  --dry-run                    Print the resolved command without executing it.
  -h, --help                   Show this help.
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

    if [[ -n "${SLURM_JOB_ID-}" ]]; then
        suffix="job${SLURM_JOB_ID}"
        if [[ -n "${SLURM_ARRAY_TASK_ID-}" ]]; then
            suffix+="_task${SLURM_ARRAY_TASK_ID}"
        fi
    else
        suffix="pid$$"
    fi

    printf '%s_%s' "$timestamp" "$suffix"
}

trim_field() {
    local value="$1"
    value="${value//$'\r'/}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
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

count_gpu_search_configs() {
    local config_path="$1"
    awk -F, '
        function trim(s) {
            gsub(/\r/, "", s)
            sub(/^[[:space:]]+/, "", s)
            sub(/[[:space:]]+$/, "", s)
            return s
        }
        {
            raw = $0
            gsub(/\r/, "", raw)
            if (raw ~ /^[[:space:]]*$/) {
                next
            }
            label = trim($1)
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

list_gpu_search_labels() {
    local config_path="$1"
    awk -F, '
        function trim(s) {
            gsub(/\r/, "", s)
            sub(/^[[:space:]]+/, "", s)
            sub(/[[:space:]]+$/, "", s)
            return s
        }
        {
            raw = $0
            gsub(/\r/, "", raw)
            if (raw ~ /^[[:space:]]*$/) {
                next
            }
            label = trim($1)
            if (label == "" || label == "label" || substr(label, 1, 1) == "#") {
                next
            }
            print label
        }
    ' "$config_path"
}

resolve_source_index_dir() {
    local dataset_dir="$1"
    local candidate=""
    local candidates=(
        "${dataset_dir}/gpu_mvr_2m"
        "${dataset_dir}/gpu_mvr_1m"
        "${dataset_dir}/gpu_mvr"
        "${dataset_dir}/gpu_mvr_index"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -d "$candidate" && -f "${candidate}/doclens.bin" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done

    return 1
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

print_hardware_info() {
    log_line "[driver] hostname=$(hostname)"
    log_line "[driver] cpu_model=$(get_cpu_model)"
    log_line "[driver] cpu_threads_available=$(nproc)"
    log_line "[driver] slurm_job_id=${SLURM_JOB_ID-<unset>}"
    log_line "[driver] slurm_array_task_id=${SLURM_ARRAY_TASK_ID-<unset>}"
    log_line "[driver] slurm_job_nodelist=${SLURM_JOB_NODELIST-<unset>}"
    log_line "[driver] slurm_cpus_per_task=${SLURM_CPUS_PER_TASK-<unset>}"
    log_line "[driver] omp_num_threads=${omp_threads}"
    log_line "[driver] cpu_bind=disabled"
}

benchmark_dataset() {
    local dataset_name="$1"
    local dataset_dir=""
    local raw_dir=""
    local source_index_dir=""
    local query_file=""
    local gt_file=""
    local impl_output_dir=""
    local impl_log_dir=""
    local run_id=""
    local results_csv=""
    local pareto_csv=""
    local log_file=""
    local config_count=""
    local config_labels=""
    local config_basename=""
    local config_stem=""
    local status=0
    local run_start_utc=""
    local run_end_utc=""
    local command=()

    validate_dataset_name "$dataset_name"

    dataset_dir="${repo_root}/dataset/${dataset_name}"
    raw_dir="${dataset_dir}/raw"
    query_file="${raw_dir}/query.bin"
    gt_file="${raw_dir}/gt.tsv"
    source_index_dir="$(resolve_source_index_dir "$dataset_dir")" || \
        die "missing index directory: expected one of ${dataset_dir}/gpu_mvr_2m, ${dataset_dir}/gpu_mvr_1m, ${dataset_dir}/gpu_mvr, or ${dataset_dir}/gpu_mvr_index"

    [[ -d "$dataset_dir" ]] || die "dataset directory not found: ${dataset_dir}"
    [[ -f "$query_file" ]] || die "missing query file: ${query_file}"
    [[ -f "$gt_file" ]] || die "missing ground truth file: ${gt_file}"
    [[ -f "${source_index_dir}/doclens.bin" ]] || die "missing index doclens file: ${source_index_dir}/doclens.bin"

    config_basename="$(basename -- "$config_file")"
    config_stem="${config_basename%.csv}"
    run_id="$(make_run_id)_${config_stem}"
    impl_output_dir="${output_dir}/${dataset_name}/${implementation_label}"
    impl_log_dir="${log_dir}/${implementation_label}/${dataset_name}"
    results_csv="${impl_output_dir}/benchmark_results_k${k}_${batch_id}.csv"
    pareto_csv="${impl_output_dir}/pareto_frontier_k${k}_${batch_id}.csv"
    log_file="${impl_log_dir}/benchmark_k${k}_${run_id}.log"
    current_log_file="$log_file"

    config_count="$(count_gpu_search_configs "$config_file")"
    config_labels="$(list_gpu_search_labels "$config_file" | paste -sd ',' -)"

    if [[ $dry_run -eq 0 ]]; then
        mkdir -p "$impl_log_dir"
        : > "$log_file"
    fi

    log_line "[driver] repo_root=${repo_root}"
    log_line "[driver] implementation=${implementation_label}"
    log_line "[driver] dataset=${dataset_name}"
    log_line "[driver] batch_id=${batch_id}"
    log_line "[driver] batch_total_config_count=${batch_total_configs}"
    log_line "[driver] run_id=${run_id}"
    log_line "[driver] config_file=${config_file}"
    log_line "[driver] config_labels=${config_labels}"
    log_line "[driver] config_count=${config_count}"
    log_line "[driver] binary=${binary}"
    log_line "[driver] query_file=${query_file}"
    log_line "[driver] gt_file=${gt_file}"
    log_line "[driver] source_index_dir=${source_index_dir}"
    log_line "[driver] index_source=nfs"
    log_line "[driver] output_dir=${output_dir}"
    log_line "[driver] results_csv=${results_csv}"
    log_line "[driver] pareto_csv=${pareto_csv}"
    log_line "[driver] log_file=${log_file}"
    log_line "[driver] parser_script=${repo_root}/script/parse_cpu_search_v3_bench_logs.py"
    log_line "[driver] k=${k}"
    log_line "[driver] nq=${nq}"
    log_line "[driver] warmup=${warmup}"
    print_hardware_info

    command=(
        env
        "OMP_NUM_THREADS=${omp_threads}"
        "$binary"
        --query "$query_file"
        --gt "$gt_file"
        --index "$source_index_dir"
        --k "$k"
        --nq "$nq"
        --warmup "$warmup"
        --config-file "$config_file"
    )

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
        die "cpu_search_v3 failed for dataset=${dataset_name}; see ${current_log_file}"
    fi

    log_line "[summary] raw_log=${log_file}"
    log_line "[summary] parse_with=$(printf '%q ' python3 "${repo_root}/script/parse_cpu_search_v3_bench_logs.py" --batch-id "${batch_id}" "${log_dir}/${implementation_label}")"
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

binary="${repo_root}/gpu-mvr/build-cpu-only/cpu_search_v3"
output_dir="${repo_root}/profiling"
log_dir="${repo_root}/slurm/logs"
implementation_label="cpu_search_v3"
k=100
nq=-1
warmup=5
omp_threads=36
dry_run=0
dataset_name=""
config_file=""
batch_id=""
batch_total_configs=0
current_log_file=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || die "missing value for --dataset"
            dataset_name="$2"
            shift 2
            ;;
        --config-file)
            [[ $# -ge 2 ]] || die "missing value for --config-file"
            config_file="$2"
            shift 2
            ;;
        --batch-id)
            [[ $# -ge 2 ]] || die "missing value for --batch-id"
            batch_id="$2"
            shift 2
            ;;
        --batch-total-configs)
            [[ $# -ge 2 ]] || die "missing value for --batch-total-configs"
            batch_total_configs="$2"
            shift 2
            ;;
        --binary)
            [[ $# -ge 2 ]] || die "missing value for --binary"
            binary="$2"
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
        --implementation-label)
            [[ $# -ge 2 ]] || die "missing value for --implementation-label"
            implementation_label="$2"
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
        --omp-threads)
            [[ $# -ge 2 ]] || die "missing value for --omp-threads"
            omp_threads="$2"
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
            if [[ -z "$dataset_name" ]]; then
                dataset_name="$1"
                shift
            else
                die "unexpected positional argument: $1"
            fi
            ;;
    esac
done

[[ -n "$dataset_name" ]] || die "--dataset is required"
[[ -n "$config_file" ]] || die "--config-file is required"
[[ -n "$batch_id" ]] || die "--batch-id is required"
validate_path_component "$implementation_label" "implementation label"
validate_path_component "$batch_id" "batch id"
require_positive_int "--k" "$k"
[[ "$nq" =~ ^-?[0-9]+$ ]] || die "--nq must be an integer"
require_positive_int "--warmup" "$warmup"
require_positive_int "--omp-threads" "$omp_threads"
require_positive_int "--batch-total-configs" "$batch_total_configs"
(( k > 0 )) || die "--k must be > 0"
(( warmup >= 0 )) || die "--warmup must be >= 0"
(( nq >= -1 )) || die "--nq must be >= -1"
(( omp_threads > 0 )) || die "--omp-threads must be > 0"
(( batch_total_configs > 0 )) || die "--batch-total-configs must be > 0"

[[ -f "$config_file" ]] || die "config CSV not found: ${config_file}"
if [[ $dry_run -eq 0 ]]; then
    [[ -x "$binary" ]] || die "search binary not found or not executable: ${binary}"
elif [[ ! -x "$binary" ]]; then
    echo "[dry-run] warning: binary not found or not executable: ${binary}" >&2
fi

benchmark_dataset "$dataset_name"

if [[ $dry_run -eq 1 ]]; then
    echo
    echo "Dry run complete."
else
    echo
    echo "Benchmarking finished."
fi
