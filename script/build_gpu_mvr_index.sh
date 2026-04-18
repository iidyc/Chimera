#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  script/build_gpu_mvr_index.sh [dataset_name] [options]
  script/build_gpu_mvr_index.sh --dataset <name> [options]

Description:
  Build a GPU-MVR index from raw dataset files using gpu-mvr/build/gpu_build_fast.
  By default the builder is executed via micromamba in the gpu-mvr env when
  micromamba is available.

Expected dataset layout:
  dataset/<name>/raw/data.bin
  dataset/<name>/raw/doclens.bin

Outputs:
  dataset/<name>/gpu_mvr_2m/     when --n-clusters is 2000000 or 2M (default)
  dataset/<name>/gpu_mvr_1m/     when --n-clusters is 1000000 or 1M
  dataset/<name>/gpu_mvr_<count>/ for other cluster counts
  log/build/<dataset>/<index_dir>/gpu_build_fast_<run_id>.log

Arguments:
  dataset_name           Dataset name under dataset/. Defaults to hotpot.
  --dataset <name>       Same as the positional dataset argument.
  --n-clusters <count>   Number of IVF/CAGRA centroids.
                         Accepts 2M=2000000 and 1M=1000000. Default: 2000000.
  --copy-raw-to-tmp      Copy raw/data.bin and raw/doclens.bin into /tmp first.
  --refresh-tmp-raw      Re-copy the /tmp raw files even if they already exist.
  --tmp-root <path>      Root directory for /tmp raw copies.
                         Default: /tmp/$USER/gpu_mvr_build
  --log-dir <path>       Root directory for raw build logs.
                         Default: log/build
  --dry-run              Print the resolved command and paths without running it.
  -h, --help             Show this help message.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

normalize_n_clusters() {
    local value="$1"
    local upper_value="${value^^}"
    case "$upper_value" in
        2M)
            printf '%s' "2000000"
            ;;
        1M)
            printf '%s' "1000000"
            ;;
        *)
            printf '%s' "$value"
            ;;
    esac
}

index_dir_name_for_n_clusters() {
    local cluster_count="$1"
    case "$cluster_count" in
        2000000)
            printf '%s' "gpu_mvr_2m"
            ;;
        1000000)
            printf '%s' "gpu_mvr_1m"
            ;;
        *)
            printf '%s' "gpu_mvr_${cluster_count}"
            ;;
    esac
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

dataset_name="hotpot"
n_clusters="2000000"
copy_raw_to_tmp=0
refresh_tmp_raw=0
tmp_root="/tmp/${USER:-user}/gpu_mvr_build"
log_dir=""
dry_run=0
dataset_set=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || die "missing value for --dataset"
            dataset_name="$2"
            dataset_set=1
            shift 2
            ;;
        --n-clusters)
            [[ $# -ge 2 ]] || die "missing value for --n-clusters"
            n_clusters="$2"
            shift 2
            ;;
        --copy-raw-to-tmp)
            copy_raw_to_tmp=1
            shift
            ;;
        --refresh-tmp-raw)
            refresh_tmp_raw=1
            shift
            ;;
        --tmp-root)
            [[ $# -ge 2 ]] || die "missing value for --tmp-root"
            tmp_root="$2"
            shift 2
            ;;
        --log-dir)
            [[ $# -ge 2 ]] || die "missing value for --log-dir"
            log_dir="$2"
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
            if [[ $dataset_set -eq 0 ]]; then
                dataset_name="$1"
                dataset_set=1
                shift
            else
                die "unexpected positional argument: $1"
            fi
            ;;
    esac
done

n_clusters="$(normalize_n_clusters "$n_clusters")"
[[ "$n_clusters" =~ ^[0-9]+$ ]] || die "--n-clusters must be a positive integer"
(( n_clusters > 0 )) || die "--n-clusters must be > 0"

case "$dataset_name" in
    ""|.|..|*/*)
        die "dataset name must be a simple directory name under dataset/"
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

gpu_build_bin="${repo_root}/gpu-mvr/build/gpu_build_fast"
micromamba_bin="$(command -v micromamba || true)"
builder_env_name="${GPU_MVR_MICROMAMBA_ENV:-gpu-mvr}"
dataset_dir="${repo_root}/dataset/${dataset_name}"
raw_dir="${dataset_dir}/raw"
data_file="${raw_dir}/data.bin"
doclens_file="${raw_dir}/doclens.bin"
index_dir_name="$(index_dir_name_for_n_clusters "$n_clusters")"
index_dir="${dataset_dir}/${index_dir_name}"
if [[ -z "$log_dir" ]]; then
    log_dir="${repo_root}/log/build"
fi

[[ -x "$gpu_build_bin" ]] || die "gpu_build binary not found or not executable: ${gpu_build_bin}"
[[ -d "$dataset_dir" ]] || die "dataset directory not found: ${dataset_dir}"
[[ -f "$data_file" ]] || die "missing embedding file: ${data_file}"
[[ -f "$doclens_file" ]] || die "missing doclens file: ${doclens_file}"

run_id="$(make_run_id)"
log_dir="${log_dir}/${dataset_name}/${index_dir_name}"
mkdir -p "$index_dir" "$log_dir"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
full_log="${log_dir}/gpu_build_fast_${run_id}.log"

if [[ $dry_run -eq 0 ]]; then
    : > "$full_log"
fi

build_cmd=(
    "$gpu_build_bin"
    --index_dir "$index_dir"
    --doclens "$doclens_file"
    --data "$data_file"
    --n_clusters "$n_clusters"
)

if [[ -n "$micromamba_bin" ]]; then
    build_cmd=(
        "$micromamba_bin"
        run
        -n "$builder_env_name"
        "${build_cmd[@]}"
    )
fi

log_capture_safe_line() {
    local line="$1"
    if [[ $dry_run -eq 1 ]]; then
        echo "$line" >&2
    else
        echo "$line" | tee -a "$full_log" >&2
    fi
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

get_cpu_threads_used() {
    if [[ -n "${OMP_NUM_THREADS-}" ]]; then
        echo "$OMP_NUM_THREADS"
        return
    fi

    nproc
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
    echo "[driver] cpu_model=$(get_cpu_model)"
    echo "[driver] cpu_threads_available=$(nproc)"
    echo "[driver] cpu_threads_used=$(get_cpu_threads_used)"
    echo "[driver] omp_num_threads=${OMP_NUM_THREADS-<unset>}"
    echo "[driver] cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
    echo "[driver] gpu_count_used=$(get_gpu_count_used)"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[driver] nvidia_smi=unavailable"
        return
    fi

    local gpu_query_output=""
    local gpu_info_lines=()
    local gpu_line=""
    if ! gpu_query_output="$(nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader 2>/dev/null)"; then
        echo "[driver] nvidia_smi_query=failed"
        return
    fi

    mapfile -t gpu_info_lines <<< "$gpu_query_output"
    echo "[driver] host_gpu_count=${#gpu_info_lines[@]}"
    for gpu_line in "${gpu_info_lines[@]}"; do
        [[ -n "$gpu_line" ]] || continue
        echo "[driver] host_gpu=${gpu_line}"
    done
}

print_run_plan() {
    cat <<EOF
[driver] repo_root=${repo_root}
[driver] dataset=${dataset_name}
[driver] micromamba_bin=${micromamba_bin:-<not-found>}
[driver] builder_env_name=${builder_env_name}
[driver] raw_dir=${raw_dir}
[driver] active_raw_dir=${active_raw_dir}
[driver] raw_source=${raw_source}
[driver] data_file=${data_file}
[driver] doclens_file=${doclens_file}
[driver] index_dir=${index_dir}
[driver] run_id=${run_id}
[driver] log_file=${full_log}
[driver] n_clusters=${n_clusters}
[driver] copy_raw_to_tmp=${copy_raw_to_tmp}
[driver] refresh_tmp_raw=${refresh_tmp_raw}
[driver] tmp_root=${tmp_root}
[driver] parser_hint=raw build log only; parse separately if needed
[driver] command=$(printf '%q ' "${build_cmd[@]}")
EOF
    print_hardware_info
}

file_size_bytes() {
    stat -c '%s' "$1"
}

tmp_raw_file_matches() {
    local source_file="$1"
    local dest_file="$2"

    [[ -f "$dest_file" ]] || return 1
    [[ "$(file_size_bytes "$source_file")" == "$(file_size_bytes "$dest_file")" ]]
}

tmp_raw_required_bytes() {
    local source_file="$1"
    local dest_file="$2"
    local source_size
    local dest_size=0

    source_size="$(file_size_bytes "$source_file")"
    if [[ -f "$dest_file" ]]; then
        dest_size="$(file_size_bytes "$dest_file")"
    fi

    if (( dest_size == source_size )); then
        echo 0
    elif (( dest_size > 0 && dest_size < source_size )); then
        echo $((source_size - dest_size))
    else
        echo "$source_size"
    fi
}

sync_tmp_raw_file() {
    local source_file="$1"
    local dest_file="$2"
    local label="$3"
    local source_size
    local dest_size=0
    local rsync_cmd=(
        rsync
        --inplace
        --partial
        --no-times
    )

    source_size="$(file_size_bytes "$source_file")"
    if [[ -f "$dest_file" ]]; then
        dest_size="$(file_size_bytes "$dest_file")"
    fi

    if (( dest_size == source_size )); then
        log_capture_safe_line "[driver] reusing_tmp_${label}=${dest_file}"
        return
    fi

    if (( dest_size > 0 && dest_size < source_size )); then
        rsync_cmd+=(--append-verify)
        log_capture_safe_line \
            "[driver] resuming_tmp_${label}=${dest_file} bytes_present=${dest_size}/${source_size}"
    else
        if (( dest_size > source_size )); then
            log_capture_safe_line \
                "[driver] replacing_tmp_${label}=${dest_file} bytes_present=${dest_size}/${source_size}"
            rm -f "$dest_file"
        else
            log_capture_safe_line "[driver] copying_tmp_${label}=${dest_file}"
        fi
    fi

    rsync_cmd+=("$source_file" "$dest_file")

    if [[ $dry_run -eq 1 ]]; then
        echo "+ $(printf '%q ' "${rsync_cmd[@]}")" >&2
        return
    fi

    echo "+ $(printf '%q ' "${rsync_cmd[@]}")" | tee -a "$full_log" >&2
    "${rsync_cmd[@]}"

    if ! tmp_raw_file_matches "$source_file" "$dest_file"; then
        die "tmp ${label} copy incomplete after rsync: ${dest_file}"
    fi
}

stage_raw_to_tmp_if_needed() {
    local source_raw_dir="$1"
    local source_data_file="$2"
    local source_doclens_file="$3"
    local dest_raw_dir="${tmp_root}/${dataset_name}/raw"
    local dest_data_file="${dest_raw_dir}/data.bin"
    local dest_doclens_file="${dest_raw_dir}/doclens.bin"

    if [[ $copy_raw_to_tmp -eq 0 ]]; then
        printf '%s\n%s\n%s\n' "$source_raw_dir" "$source_data_file" "$source_doclens_file"
        return
    fi

    if [[ $refresh_tmp_raw -eq 0 ]] \
        && tmp_raw_file_matches "$source_data_file" "$dest_data_file" \
        && tmp_raw_file_matches "$source_doclens_file" "$dest_doclens_file"; then
        log_capture_safe_line "[driver] reusing_tmp_raw=${dest_raw_dir}"
        printf '%s\n%s\n%s\n' "$dest_raw_dir" "$dest_data_file" "$dest_doclens_file"
        return
    fi

    log_capture_safe_line "[driver] tmp_raw_source=${source_raw_dir}"
    log_capture_safe_line "[driver] tmp_raw_target=${dest_raw_dir}"

    if [[ $dry_run -eq 1 ]]; then
        if [[ $refresh_tmp_raw -eq 1 ]]; then
            echo "+ rm -rf ${dest_raw_dir}" >&2
        fi
        echo "+ mkdir -p ${dest_raw_dir}" >&2
        sync_tmp_raw_file "$source_data_file" "$dest_data_file" "data_file"
        sync_tmp_raw_file "$source_doclens_file" "$dest_doclens_file" "doclens_file"
        printf '%s\n%s\n%s\n' "$dest_raw_dir" "$dest_data_file" "$dest_doclens_file"
        return
    fi

    local parent_dir
    parent_dir="$(dirname "$dest_raw_dir")"
    mkdir -p "$parent_dir"

    local required_bytes
    local available_bytes
    required_bytes=0
    if [[ $refresh_tmp_raw -eq 1 ]]; then
        required_bytes="$(( $(file_size_bytes "$source_data_file") + $(file_size_bytes "$source_doclens_file") ))"
    else
        required_bytes="$(( \
            $(tmp_raw_required_bytes "$source_data_file" "$dest_data_file") + \
            $(tmp_raw_required_bytes "$source_doclens_file" "$dest_doclens_file") \
        ))"
    fi
    available_bytes="$(df -B1 "$parent_dir" | awk 'NR==2 {print $4}')"
    if (( available_bytes < required_bytes )); then
        die "not enough free space under ${parent_dir} to copy raw files"
    fi

    local copy_start_utc
    local copy_end_utc
    local copy_start_s
    local copy_end_s
    copy_start_utc="$(date -u +%Y%m%dT%H%M%SZ)"
    copy_start_s="$(date +%s)"
    log_capture_safe_line "[driver] tmp_copy_start_utc=${copy_start_utc}"

    if [[ $refresh_tmp_raw -eq 1 ]]; then
        echo "+ $(printf '%q ' rm -rf "$dest_raw_dir")" | tee -a "$full_log" >&2
        rm -rf "$dest_raw_dir"
    fi
    echo "+ $(printf '%q ' mkdir -p "$dest_raw_dir")" | tee -a "$full_log" >&2
    mkdir -p "$dest_raw_dir"
    sync_tmp_raw_file "$source_data_file" "$dest_data_file" "data_file"
    sync_tmp_raw_file "$source_doclens_file" "$dest_doclens_file" "doclens_file"

    copy_end_utc="$(date -u +%Y%m%dT%H%M%SZ)"
    copy_end_s="$(date +%s)"
    log_capture_safe_line "[driver] tmp_copy_end_utc=${copy_end_utc}"
    log_capture_safe_line "[driver] tmp_copy_elapsed_seconds=$((copy_end_s - copy_start_s))"
    log_capture_safe_line "[driver] tmp_raw_size=$(du -sh "$dest_raw_dir" | awk '{print $1}')"

    printf '%s\n%s\n%s\n' "$dest_raw_dir" "$dest_data_file" "$dest_doclens_file"
}

raw_source="nfs"
active_raw_dir="$raw_dir"
staged_paths=()

mapfile -t staged_paths < <(stage_raw_to_tmp_if_needed "$raw_dir" "$data_file" "$doclens_file")
if [[ ${#staged_paths[@]} -eq 3 ]]; then
    active_raw_dir="${staged_paths[0]}"
    data_file="${staged_paths[1]}"
    doclens_file="${staged_paths[2]}"
else
    die "failed to resolve active raw paths"
fi

if [[ "$active_raw_dir" != "$raw_dir" ]]; then
    raw_source="tmp"
fi

build_cmd=(
    "$gpu_build_bin"
    --index_dir "$index_dir"
    --doclens "$doclens_file"
    --data "$data_file"
    --n_clusters "$n_clusters"
)

if [[ -n "$micromamba_bin" ]]; then
    build_cmd=(
        "$micromamba_bin"
        run
        -n "$builder_env_name"
        "${build_cmd[@]}"
    )
fi

if [[ $dry_run -eq 1 ]]; then
    print_run_plan
    exit 0
fi

print_run_plan | tee -a "$full_log"
echo "[driver] build_start_utc=${timestamp}" | tee -a "$full_log"

run_build() {
    local command_prefix=()
    if command -v stdbuf >/dev/null 2>&1; then
        command_prefix=(stdbuf -oL -eL)
    fi

    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v "${command_prefix[@]}" "${build_cmd[@]}"
    else
        "${command_prefix[@]}" "${build_cmd[@]}"
    fi
}

set +e
run_build 2>&1 | tee -a "$full_log"
build_status=${PIPESTATUS[0]}
set -e

end_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
echo "[driver] build_end_utc=${end_timestamp}" | tee -a "$full_log"
echo "[driver] build_exit_code=${build_status}" | tee -a "$full_log"

if [[ $build_status -ne 0 ]]; then
    echo "[driver] build failed; see ${full_log}" >&2
    exit "$build_status"
fi

echo "[driver] build completed successfully" | tee -a "$full_log"
