#!/usr/bin/env bash
#SBATCH --job-name=cpu-search-v3-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mincpus=72
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --output=slurm/logs/slurm_%x_%A_%a.out
#SBATCH --error=slurm/logs/slurm_%x_%A_%a.err

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${REPO_ROOT:-$(cd -- "${script_dir}/.." && pwd)}"
manifest="${CPU_SEARCH_V3_PROFILE_MANIFEST:?CPU_SEARCH_V3_PROFILE_MANIFEST is required}"
batch_id="${CPU_SEARCH_V3_PROFILE_BATCH_ID:?CPU_SEARCH_V3_PROFILE_BATCH_ID is required}"
binary="${CPU_SEARCH_V3_PROFILE_BINARY:?CPU_SEARCH_V3_PROFILE_BINARY is required}"
log_dir="${CPU_SEARCH_V3_PROFILE_LOG_DIR:?CPU_SEARCH_V3_PROFILE_LOG_DIR is required}"
k_default="${CPU_SEARCH_V3_PROFILE_K:?CPU_SEARCH_V3_PROFILE_K is required}"
nq="${CPU_SEARCH_V3_PROFILE_NQ:?CPU_SEARCH_V3_PROFILE_NQ is required}"
warmup="${CPU_SEARCH_V3_PROFILE_WARMUP:?CPU_SEARCH_V3_PROFILE_WARMUP is required}"
omp_threads="${CPU_SEARCH_V3_PROFILE_OMP_THREADS:?CPU_SEARCH_V3_PROFILE_OMP_THREADS is required}"

line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$manifest")"
IFS=$'\t' read -r dataset_name config_file k selection_label target_recall <<< "$line"
k="${k:-$k_default}"

cmd=(
    bash
    "${repo_root}/slurm/run_cpu_search_v3_profile_chunk.sh"
    --dataset "$dataset_name"
    --config-file "$config_file"
    --batch-id "$batch_id"
    --selection-label "$selection_label"
    --target-recall "$target_recall"
    --binary "$binary"
    --log-dir "$log_dir"
    --k "$k"
    --nq "$nq"
    --warmup "$warmup"
    --omp-threads "$omp_threads"
)

echo "[slurm] task_id=${SLURM_ARRAY_TASK_ID}"
echo "[slurm] dataset=${dataset_name}"
echo "[slurm] config_file=${config_file}"
echo "[slurm] k=${k}"
echo "[slurm] selection_label=${selection_label}"
echo "[slurm] command=$(printf '%q ' "${cmd[@]}")"
"${cmd[@]}"
