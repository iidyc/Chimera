#!/usr/bin/env bash

set -euo pipefail

remote_host="${REMOTE_HOST:-vast}"
remote_root="${REMOTE_ROOT:-/workspace}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dry_run="${DRY_RUN:-0}"

rsync_args=(
    -av
    --whole-file
    --human-readable
    --info=progress2
)

if [[ "${dry_run}" == "1" ]]; then
    rsync_args+=(--dry-run)
fi

run_rsync() {
    rsync "${rsync_args[@]}" "$@"
}

cd "${repo_root}"

if [[ "${dry_run}" == "1" ]]; then
    echo "[dry-run] ssh ${remote_host} mkdir -p ${remote_root}"
else
    ssh "${remote_host}" "mkdir -p '${remote_root}'"
fi

# Runnable GPU-MVR source, scripts, and small configuration files.
run_rsync \
    --exclude='**/build/' \
    --exclude='**/build-*/' \
    --exclude='**/__pycache__/' \
    --exclude='*.pyc' \
    gpu-mvr/ \
    "${remote_host}:${remote_root}/gpu-mvr/"

run_rsync \
    --prune-empty-dirs \
    --include='*/' \
    --include='bench_gpu_mvr.sh' \
    --include='bench_all_gpu_mvr_versions.sh' \
    --include='build_gpu_mvr_index.sh' \
    --include='build_all_gpu_mvr_indices.sh' \
    --include='cmake.sh' \
    --include='compare_benchmark_pareto.py' \
    --include='compare_v6lite_v7lite_v8_slots_recall_targets.py' \
    --include='convert_gpu_mvr_index_layout.py' \
    --include='gpu_mvr_experiment_lib.py' \
    --include='parse_bench_logs.py' \
    --include='parse_cpu_search_v3_bench_logs.py' \
    --include='parse_cpu_search_v3_profile_logs.py' \
    --include='plot_cpu_search_v3_stage_breakdown.py' \
    --include='run_gpu_*.py' \
    --include='setup_gpu_mvr_env.sh' \
    --include='select_cpu_search_v3_best_configs.py' \
    --exclude='**/__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*' \
    script/ \
    "${remote_host}:${remote_root}/script/"

if [[ -d slurm ]]; then
    run_rsync slurm/ "${remote_host}:${remote_root}/slurm/"
fi

if [[ -d profiling ]]; then
    run_rsync \
        --prune-empty-dirs \
        --include='*/' \
        --include='*.csv' \
        --exclude='*' \
        profiling/ \
        "${remote_host}:${remote_root}/profiling/"
fi

# Runtime data needed by GPU-MVR search/benchmark scripts:
#   - dataset/<name>/gpu_mvr_2m/ index files
#   - dataset/<name>/raw/query.bin
#   - dataset/<name>/raw/gt.tsv
#   - dataset/<name>/raw/doclens.bin when present
# Deliberately excludes dataset/<name>/raw/data.bin and other raw corpus files.
run_rsync \
    --prune-empty-dirs \
    --include='*/' \
    --include='*/gpu_mvr_2m/***' \
    --include='*/raw/query.bin' \
    --include='*/raw/gt.tsv' \
    --include='*/raw/doclens.bin' \
    --exclude='*/raw/data.bin' \
    --exclude='*/raw/***' \
    --exclude='***' \
    dataset/ \
    "${remote_host}:${remote_root}/dataset/"

cat <<EOF
Migration complete.
Remote project root: ${remote_host}:${remote_root}

Override defaults with:
  REMOTE_HOST=vast REMOTE_ROOT=/workspace DRY_RUN=1 ./$(basename "$0")
EOF
