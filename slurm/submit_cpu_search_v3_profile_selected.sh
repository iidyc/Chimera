#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  slurm/submit_cpu_search_v3_profile_selected.sh --selection-csv <path> --batch-id <id> [options]
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

selection_csv=""
batch_id=""
binary="${repo_root}/gpu-mvr/build-cpu-only/cpu_search_v3"
log_dir="${repo_root}/slurm/logs/cpu_search_v3_profile"
generated_dir="${repo_root}/slurm/generated"
partition=""
time_limit="04:00:00"
mem="128G"
dependency=""
k_default=100
nq=-1
warmup=5
omp_threads=36

while [[ $# -gt 0 ]]; do
    case "$1" in
        --selection-csv) selection_csv="$2"; shift 2 ;;
        --batch-id) batch_id="$2"; shift 2 ;;
        --binary) binary="$2"; shift 2 ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        --generated-dir) generated_dir="$2"; shift 2 ;;
        --partition) partition="$2"; shift 2 ;;
        --time) time_limit="$2"; shift 2 ;;
        --mem) mem="$2"; shift 2 ;;
        --dependency) dependency="$2"; shift 2 ;;
        --k-default) k_default="$2"; shift 2 ;;
        --nq) nq="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --omp-threads) omp_threads="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n "$selection_csv" ]] || die "--selection-csv is required"
[[ -n "$batch_id" ]] || die "--batch-id is required"
[[ -x "$binary" ]] || die "binary not found or not executable: ${binary}"
[[ -f "$selection_csv" ]] || die "selection CSV not found: ${selection_csv}"

batch_generated_dir="${generated_dir}/cpu_search_v3_profile/${batch_id}"
mkdir -p "$batch_generated_dir"
manifest="${batch_generated_dir}/manifest.tsv"

{
    tr -d '\r' < "$selection_csv" | tail -n +2 | while IFS=, read -r dataset k target_recall label _recall _qps _avg_latency _p50 _p90 _p95 _p99 _max _stddev _queries _warmup _nprobe _k_rank_cluster _k_rank_all_tokens _itopk_size _overlap_chunks _results_csv config_csv; do
        printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$config_csv" "$k" "$label" "$target_recall"
    done
} > "$manifest"

task_count="$(wc -l < "$manifest" | tr -d '[:space:]')"
[[ "$task_count" != "0" ]] || die "selection CSV produced an empty manifest"

export_vars=(
    "ALL"
    "REPO_ROOT=${repo_root}"
    "CPU_SEARCH_V3_PROFILE_MANIFEST=${manifest}"
    "CPU_SEARCH_V3_PROFILE_BATCH_ID=${batch_id}"
    "CPU_SEARCH_V3_PROFILE_BINARY=${binary}"
    "CPU_SEARCH_V3_PROFILE_LOG_DIR=${log_dir}"
    "CPU_SEARCH_V3_PROFILE_K=${k_default}"
    "CPU_SEARCH_V3_PROFILE_NQ=${nq}"
    "CPU_SEARCH_V3_PROFILE_WARMUP=${warmup}"
    "CPU_SEARCH_V3_PROFILE_OMP_THREADS=${omp_threads}"
)

sbatch_cmd=(
    sbatch
    --parsable
    --chdir "$repo_root"
    --job-name "cpu_search_v3-profile-${batch_id}"
    --export "$(IFS=,; echo "${export_vars[*]}")"
    --array "0-$((task_count - 1))"
    --mem "$mem"
)
if [[ -n "$partition" ]]; then
    sbatch_cmd+=(--partition "$partition")
fi
if [[ -n "$time_limit" ]]; then
    sbatch_cmd+=(--time "$time_limit")
fi
if [[ -n "$dependency" ]]; then
    sbatch_cmd+=(--dependency "$dependency")
fi
sbatch_cmd+=("${repo_root}/slurm/cpu_search_v3_profile_array.sbatch")

echo "[submit] batch_id=${batch_id}"
echo "[submit] selection_csv=${selection_csv}"
echo "[submit] manifest=${manifest}"
echo "[submit] task_count=${task_count}"
echo "[submit] sbatch_command=$(printf '%q ' "${sbatch_cmd[@]}")"
job_id="$("${sbatch_cmd[@]}")"
echo "[submit] slurm_job_id=${job_id}"
