#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  slurm/run_cpu_search_v3_profile_chunk.sh --dataset <name> --config-file <path> --batch-id <id> [options]

Description:
  Run cpu_search_v3 with --profile-eval-all-queries for one selected config and
  write one aggregate profile log for that dataset/config pair.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

log_line() {
    local line="$1"
    echo "$line" | tee -a "$current_log_file"
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

validate_dataset_name() {
    case "$1" in
        ""|.|..|*/*)
            die "dataset name must be a simple directory name under dataset/: $1"
            ;;
    esac
}

require_positive_int() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer"
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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

dataset_name=""
config_file=""
batch_id=""
selection_label=""
target_recall=""
binary="${repo_root}/gpu-mvr/build-cpu-only/cpu_search_v3"
log_dir="${repo_root}/slurm/logs/cpu_search_v3_profile"
k=100
nq=-1
warmup=5
omp_threads=36

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) dataset_name="$2"; shift 2 ;;
        --config-file) config_file="$2"; shift 2 ;;
        --batch-id) batch_id="$2"; shift 2 ;;
        --selection-label) selection_label="$2"; shift 2 ;;
        --target-recall) target_recall="$2"; shift 2 ;;
        --binary) binary="$2"; shift 2 ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        --k) k="$2"; shift 2 ;;
        --nq) nq="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --omp-threads) omp_threads="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n "$dataset_name" ]] || die "--dataset is required"
[[ -n "$config_file" ]] || die "--config-file is required"
[[ -n "$batch_id" ]] || die "--batch-id is required"
[[ -n "$selection_label" ]] || die "--selection-label is required"
[[ -n "$target_recall" ]] || die "--target-recall is required"
validate_dataset_name "$dataset_name"
require_positive_int "--k" "$k"
require_positive_int "--warmup" "$warmup"
require_positive_int "--omp-threads" "$omp_threads"
[[ "$nq" =~ ^-?[0-9]+$ ]] || die "--nq must be an integer"
[[ -x "$binary" ]] || die "binary not found or not executable: ${binary}"
[[ -f "$config_file" ]] || die "config CSV not found: ${config_file}"

dataset_dir="${repo_root}/dataset/${dataset_name}"
raw_dir="${dataset_dir}/raw"
query_file="${raw_dir}/query.bin"
gt_file="${raw_dir}/gt.tsv"
source_index_dir="$(resolve_source_index_dir "$dataset_dir")" || die "missing source index dir for ${dataset_name}"
run_id="$(make_run_id)"
dataset_log_dir="${log_dir}/${dataset_name}"
mkdir -p "$dataset_log_dir"
current_log_file="${dataset_log_dir}/profile_k${k}_${run_id}_${selection_label}.log"
: > "$current_log_file"

log_line "[driver] repo_root=${repo_root}"
log_line "[driver] implementation=cpu_search_v3"
log_line "[driver] dataset=${dataset_name}"
log_line "[driver] batch_id=${batch_id}"
log_line "[driver] selection_label=${selection_label}"
log_line "[driver] target_recall=${target_recall}"
log_line "[driver] config_file=${config_file}"
log_line "[driver] binary=${binary}"
log_line "[driver] query_file=${query_file}"
log_line "[driver] gt_file=${gt_file}"
log_line "[driver] source_index_dir=${source_index_dir}"
log_line "[driver] k=${k}"
log_line "[driver] nq=${nq}"
log_line "[driver] warmup=${warmup}"
log_line "[driver] omp_num_threads=${omp_threads}"
log_line "[driver] cpu_bind=disabled"
log_line "[driver] log_file=${current_log_file}"
log_line "[driver] parser_script=${repo_root}/script/parse_cpu_search_v3_profile_logs.py"

cmd=(
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
    --profile-eval-all-queries
)

log_line "[run] command=$(printf '%q ' "${cmd[@]}")"
log_line "[run] start_utc=$(date -u +%Y%m%dT%H%M%SZ)"
set +e
stdbuf -oL -eL "${cmd[@]}" 2>&1 | tee -a "$current_log_file"
status=${PIPESTATUS[0]}
set -e
log_line "[run] end_utc=$(date -u +%Y%m%dT%H%M%SZ)"
log_line "[run] exit_code=${status}"
if [[ $status -ne 0 ]]; then
    die "profile run failed; see ${current_log_file}"
fi
