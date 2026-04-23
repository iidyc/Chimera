#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  slurm/submit_cpu_search_v3_bench.sh [--dataset <name> ...] [options]
  slurm/submit_cpu_search_v3_bench.sh [dataset_name ...] [options]

Description:
  Split profiling/gpu_mvr_config.csv into smaller config chunks and submit a
  Slurm array so each task benchmarks one dataset with one config chunk.

  Defaults:
    datasets: lotte hotpot msmarco
    binary:   gpu-mvr/build-cpu-only/cpu_search_v3
    OMP:      36 threads
    memory:   128G
    node filter: 72-CPU machines only

Options:
  --dataset <name>             Dataset under dataset/. Repeatable.
                               Accepts hotspot as an alias for hotpot.
  --config-file <path>         Config CSV to split. Default: profiling/gpu_mvr_config.csv
  --configs-per-job <n>        Number of configs in each Slurm task. Default: 1
  --binary <path>              Search binary. Default: gpu-mvr/build-cpu-only/cpu_search_v3
  --output-dir <path>          Combined CSV output root. Default: profiling
  --log-dir <path>             Slurm and raw benchmark log root. Default: slurm/logs
  --generated-dir <path>       Generated manifest/chunk root. Default: slurm/generated
  --implementation-label <id>  Output folder label. Default: cpu_search_v3
  --batch-id <id>              Override the generated batch identifier.
  --partition <name>           Passed through to sbatch.
  --time <limit>               Passed through to sbatch, e.g. 08:00:00.
  --mem <size>                 Override the sbatch memory request. Default: 128G
  --max-concurrent <n>         Limit concurrent array tasks with %n.
  --k <top_k>                  Final retrieval depth / recall depth. Default: 100
  --nq <count>                 Evaluation queries timed after warmup. Default: -1
  --warmup <count>             Warmup queries before timed evaluation. Default: 5
  --omp-threads <count>        OMP_NUM_THREADS value. Default: 36
  --dry-run                    Build manifests and print sbatch command without submitting.
  -h, --help                   Show this help.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

trim_field() {
    local value="$1"
    value="${value//$'\r'/}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
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

make_batch_id() {
    printf '%s' "$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
}

load_config_rows() {
    local source_config="$1"
    config_header=""
    config_rows=()

    while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
        raw_line="${raw_line%$'\r'}"
        [[ "$raw_line" =~ ^[[:space:]]*$ ]] && continue

        IFS=',' read -r first_field _ <<< "$raw_line"
        first_field="$(trim_field "$first_field")"
        [[ -z "$first_field" ]] && continue
        [[ "${first_field:0:1}" == "#" ]] && continue

        if [[ -z "$config_header" ]]; then
            if [[ "$raw_line" == *label* && "$raw_line" == *nprobe* && "$raw_line" == *k_rank_cluster* && "$raw_line" == *k_rank_all_tokens* && "$raw_line" == *itopk_size* && "$raw_line" == *overlap_chunks* ]]; then
                config_header="$raw_line"
                continue
            fi
        fi

        if [[ "$first_field" == "label" ]]; then
            config_header="$raw_line"
            continue
        fi

        config_rows+=("$raw_line")
    done < "$source_config"

    ((${#config_rows[@]} > 0)) || die "no benchmark configs found in ${source_config}"
}

split_config_chunks() {
    local source_config="$1"
    local dest_dir="$2"
    local chunk_size="$3"
    local chunk_index=0
    local row_index=0
    local total_rows="${#config_rows[@]}"

    chunk_files=()

    while (( row_index < total_rows )); do
        local chunk_file
        chunk_file="$(printf '%s/config_chunk_%03d.csv' "$dest_dir" "$((chunk_index + 1))")"
        chunk_files+=("$chunk_file")

        if [[ -n "$config_header" ]]; then
            printf '%s\n' "$config_header" > "$chunk_file"
        else
            : > "$chunk_file"
        fi

        local chunk_end=$(( row_index + chunk_size ))
        while (( row_index < chunk_end && row_index < total_rows )); do
            printf '%s\n' "${config_rows[row_index]}" >> "$chunk_file"
            row_index=$((row_index + 1))
        done

        chunk_index=$((chunk_index + 1))
    done
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

datasets=(lotte hotpot msmarco)
dataset_set=0
config_file="${repo_root}/profiling/gpu_mvr_config.csv"
configs_per_job=1
binary="${repo_root}/gpu-mvr/build-cpu-only/cpu_search_v3"
output_dir="${repo_root}/profiling"
log_dir="${repo_root}/slurm/logs"
generated_dir="${repo_root}/slurm/generated"
implementation_label="cpu_search_v3"
batch_id=""
partition=""
time_limit=""
mem="128G"
max_concurrent=0
k=100
nq=-1
warmup=5
omp_threads=36
dry_run=0

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
        --config-file)
            [[ $# -ge 2 ]] || die "missing value for --config-file"
            config_file="$2"
            shift 2
            ;;
        --configs-per-job)
            [[ $# -ge 2 ]] || die "missing value for --configs-per-job"
            configs_per_job="$2"
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
        --generated-dir)
            [[ $# -ge 2 ]] || die "missing value for --generated-dir"
            generated_dir="$2"
            shift 2
            ;;
        --implementation-label)
            [[ $# -ge 2 ]] || die "missing value for --implementation-label"
            implementation_label="$2"
            shift 2
            ;;
        --batch-id)
            [[ $# -ge 2 ]] || die "missing value for --batch-id"
            batch_id="$2"
            shift 2
            ;;
        --partition)
            [[ $# -ge 2 ]] || die "missing value for --partition"
            partition="$2"
            shift 2
            ;;
        --time)
            [[ $# -ge 2 ]] || die "missing value for --time"
            time_limit="$2"
            shift 2
            ;;
        --mem)
            [[ $# -ge 2 ]] || die "missing value for --mem"
            mem="$2"
            shift 2
            ;;
        --max-concurrent)
            [[ $# -ge 2 ]] || die "missing value for --max-concurrent"
            max_concurrent="$2"
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
            normalized_dataset="$(normalize_dataset_name "$1")"
            if [[ $dataset_set -eq 0 ]]; then
                datasets=()
                dataset_set=1
            fi
            if ! append_unique "$normalized_dataset" "${datasets[@]}"; then
                datasets+=("$normalized_dataset")
            fi
            shift
            ;;
    esac
done

[[ ${#datasets[@]} -gt 0 ]] || die "no datasets selected"
for dataset_name in "${datasets[@]}"; do
    validate_dataset_name "$dataset_name"
done

validate_path_component "$implementation_label" "implementation label"
if [[ -z "$batch_id" ]]; then
    batch_id="$(make_batch_id)"
fi
validate_path_component "$batch_id" "batch id"
require_positive_int "--configs-per-job" "$configs_per_job"
require_positive_int "--k" "$k"
[[ "$nq" =~ ^-?[0-9]+$ ]] || die "--nq must be an integer"
require_positive_int "--warmup" "$warmup"
require_positive_int "--omp-threads" "$omp_threads"
require_positive_int "--max-concurrent" "$max_concurrent"
(( configs_per_job > 0 )) || die "--configs-per-job must be > 0"
(( k > 0 )) || die "--k must be > 0"
(( warmup >= 0 )) || die "--warmup must be >= 0"
(( nq >= -1 )) || die "--nq must be >= -1"
(( omp_threads > 0 )) || die "--omp-threads must be > 0"

[[ -f "$config_file" ]] || die "config CSV not found: ${config_file}"
if [[ $dry_run -eq 0 ]]; then
    [[ -x "$binary" ]] || die "search binary not found or not executable: ${binary}"
elif [[ ! -x "$binary" ]]; then
    echo "[dry-run] warning: binary not found or not executable: ${binary}" >&2
fi

mkdir -p "$log_dir" "$generated_dir"

batch_generated_dir="${generated_dir}/${implementation_label}/${batch_id}"
[[ ! -e "$batch_generated_dir" ]] || die "generated batch directory already exists: ${batch_generated_dir}"
mkdir -p "$batch_generated_dir"

chunks_dir="${batch_generated_dir}/config_chunks"
mkdir -p "$chunks_dir"

load_config_rows "$config_file"
total_configs="${#config_rows[@]}"
split_config_chunks "$config_file" "$chunks_dir" "$configs_per_job"

manifest_file="${batch_generated_dir}/manifest.tsv"
: > "$manifest_file"
for dataset_name in "${datasets[@]}"; do
    for chunk_file in "${chunk_files[@]}"; do
        printf '%s\t%s\n' "$dataset_name" "$chunk_file" >> "$manifest_file"
    done
done

task_count=$(( ${#datasets[@]} * ${#chunk_files[@]} ))
(( task_count > 0 )) || die "no Slurm tasks were generated"

array_spec="0-$((task_count - 1))"
if (( max_concurrent > 0 )); then
    array_spec="${array_spec}%${max_concurrent}"
fi

export_vars=(
    "ALL"
    "REPO_ROOT=${repo_root}"
    "CPU_SEARCH_V3_MANIFEST=${manifest_file}"
    "CPU_SEARCH_V3_BATCH_ID=${batch_id}"
    "CPU_SEARCH_V3_TOTAL_CONFIGS=${total_configs}"
    "CPU_SEARCH_V3_BINARY=${binary}"
    "CPU_SEARCH_V3_OUTPUT_DIR=${output_dir}"
    "CPU_SEARCH_V3_LOG_DIR=${log_dir}"
    "CPU_SEARCH_V3_IMPL_LABEL=${implementation_label}"
    "CPU_SEARCH_V3_K=${k}"
    "CPU_SEARCH_V3_NQ=${nq}"
    "CPU_SEARCH_V3_WARMUP=${warmup}"
    "CPU_SEARCH_V3_OMP_THREADS=${omp_threads}"
)

sbatch_cmd=(
    sbatch
    --parsable
    --chdir "$repo_root"
    --job-name "${implementation_label}-k${k}-${batch_id}"
    --export "$(IFS=,; echo "${export_vars[*]}")"
    --array "$array_spec"
    --mem "$mem"
)

if [[ -n "$partition" ]]; then
    sbatch_cmd+=(--partition "$partition")
fi
if [[ -n "$time_limit" ]]; then
    sbatch_cmd+=(--time "$time_limit")
fi

sbatch_cmd+=("${repo_root}/slurm/cpu_search_v3_array.sbatch")

echo "[submit] batch_id=${batch_id}"
echo "[submit] datasets=${datasets[*]}"
echo "[submit] config_file=${config_file}"
echo "[submit] total_configs=${total_configs}"
echo "[submit] configs_per_job=${configs_per_job}"
echo "[submit] chunk_count=${#chunk_files[@]}"
echo "[submit] task_count=${task_count}"
echo "[submit] manifest=${manifest_file}"
echo "[submit] generated_dir=${batch_generated_dir}"
echo "[submit] binary=${binary}"
echo "[submit] output_dir=${output_dir}"
echo "[submit] log_dir=${log_dir}"
echo "[submit] parse_command=$(printf '%q ' python3 "${repo_root}/script/parse_cpu_search_v3_bench_logs.py" --batch-id "${batch_id}" "${log_dir}/${implementation_label}")"
echo "[submit] sbatch_command=$(printf '%q ' "${sbatch_cmd[@]}")"

if [[ $dry_run -eq 1 ]]; then
    echo
    echo "Dry run complete."
    exit 0
fi

job_id="$("${sbatch_cmd[@]}")"
echo "[submit] slurm_job_id=${job_id}"
