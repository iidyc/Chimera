#!/usr/bin/env bash
#SBATCH --job-name=build-cpu-search-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mincpus=72
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/slurm_%x_%A.out
#SBATCH --error=slurm/logs/slurm_%x_%A.err

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "${script_dir}/.." && pwd)}}"
build_dir="${CPU_SEARCH_V3_BUILD_DIR:-${repo_root}/gpu-mvr/build-cpu-only}"
target="${CPU_SEARCH_V3_BUILD_TARGET:-cpu_search_v3}"

echo "[build] repo_root=${repo_root}"
echo "[build] build_dir=${build_dir}"
echo "[build] target=${target}"
cmake --build "${build_dir}" --target "${target}" -j "${SLURM_CPUS_PER_TASK:-8}"
