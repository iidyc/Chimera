#!/usr/bin/env bash
#SBATCH --job-name=cpu-search-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mincpus=72
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --output=slurm/logs/slurm_%x_%A_%a.out
#SBATCH --error=slurm/logs/slurm_%x_%A_%a.err

set -euo pipefail

die() {
    echo "error: $*" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${REPO_ROOT:-$(cd -- "${script_dir}/.." && pwd)}"

manifest="${CPU_SEARCH_V3_MANIFEST:?CPU_SEARCH_V3_MANIFEST is required}"
batch_id="${CPU_SEARCH_V3_BATCH_ID:?CPU_SEARCH_V3_BATCH_ID is required}"
total_configs="${CPU_SEARCH_V3_TOTAL_CONFIGS:?CPU_SEARCH_V3_TOTAL_CONFIGS is required}"
binary="${CPU_SEARCH_V3_BINARY:?CPU_SEARCH_V3_BINARY is required}"
output_dir="${CPU_SEARCH_V3_OUTPUT_DIR:?CPU_SEARCH_V3_OUTPUT_DIR is required}"
log_dir="${CPU_SEARCH_V3_LOG_DIR:?CPU_SEARCH_V3_LOG_DIR is required}"
implementation_label="${CPU_SEARCH_V3_IMPL_LABEL:?CPU_SEARCH_V3_IMPL_LABEL is required}"
k="${CPU_SEARCH_V3_K:?CPU_SEARCH_V3_K is required}"
nq="${CPU_SEARCH_V3_NQ:?CPU_SEARCH_V3_NQ is required}"
warmup="${CPU_SEARCH_V3_WARMUP:?CPU_SEARCH_V3_WARMUP is required}"
omp_threads="${CPU_SEARCH_V3_OMP_THREADS:?CPU_SEARCH_V3_OMP_THREADS is required}"

[[ -f "$manifest" ]] || die "manifest file not found: ${manifest}"

manifest_line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$manifest")"
[[ -n "$manifest_line" ]] || die "no manifest entry for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

IFS=$'\t' read -r dataset_name config_file <<< "$manifest_line"
[[ -n "$dataset_name" ]] || die "manifest entry is missing dataset for task ${SLURM_ARRAY_TASK_ID}"
[[ -n "$config_file" ]] || die "manifest entry is missing config file for task ${SLURM_ARRAY_TASK_ID}"

cmd=(
    "${repo_root}/slurm/run_cpu_search_v3_chunk.sh"
    --dataset "$dataset_name"
    --config-file "$config_file"
    --batch-id "$batch_id"
    --batch-total-configs "$total_configs"
    --binary "$binary"
    --output-dir "$output_dir"
    --log-dir "$log_dir"
    --implementation-label "$implementation_label"
    --k "$k"
    --nq "$nq"
    --warmup "$warmup"
    --omp-threads "$omp_threads"
)

echo "[slurm] task_id=${SLURM_ARRAY_TASK_ID}"
echo "[slurm] dataset=${dataset_name}"
echo "[slurm] config_file=${config_file}"
echo "[slurm] command=$(printf '%q ' "${cmd[@]}")"

"${cmd[@]}"
