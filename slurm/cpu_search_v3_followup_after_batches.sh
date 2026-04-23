#!/usr/bin/env bash

set -euo pipefail

die() {
    echo "error: $*" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

k10_batch_id=""
k100_batch_id=""
profile_batch_id=""
profile_build_job_id=""
implementation_label="cpu_search_v3"
results_root="${repo_root}/profiling"
generated_dir="${repo_root}/slurm/generated"
profile_log_dir="${repo_root}/slurm/logs/cpu_search_v3_profile"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --k10-batch-id) k10_batch_id="$2"; shift 2 ;;
        --k100-batch-id) k100_batch_id="$2"; shift 2 ;;
        --profile-batch-id) profile_batch_id="$2"; shift 2 ;;
        --profile-build-job-id) profile_build_job_id="$2"; shift 2 ;;
        --implementation-label) implementation_label="$2"; shift 2 ;;
        --results-root) results_root="$2"; shift 2 ;;
        --generated-dir) generated_dir="$2"; shift 2 ;;
        --profile-log-dir) profile_log_dir="$2"; shift 2 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n "$k10_batch_id" ]] || die "--k10-batch-id is required"
[[ -n "$k100_batch_id" ]] || die "--k100-batch-id is required"
[[ -n "$profile_batch_id" ]] || die "--profile-batch-id is required"
[[ -n "$profile_build_job_id" ]] || die "--profile-build-job-id is required"

python3 "${repo_root}/script/parse_cpu_search_v3_bench_logs.py" \
    --repo-root "$repo_root" \
    --batch-id "$k10_batch_id" \
    "${repo_root}/slurm/logs/cpu_search_v3"
python3 "${repo_root}/script/parse_cpu_search_v3_bench_logs.py" \
    --repo-root "$repo_root" \
    --batch-id "$k100_batch_id" \
    "${repo_root}/slurm/logs/cpu_search_v3"

selection_dir="${generated_dir}/cpu_search_v3_followup/${profile_batch_id}"
mkdir -p "$selection_dir"
selection_csv="${selection_dir}/selected_configs.csv"
config_dir="${selection_dir}/configs"

python3 "${repo_root}/script/select_cpu_search_v3_best_configs.py" \
    --results-root "$results_root" \
    --implementation-label "$implementation_label" \
    --k10-batch-id "$k10_batch_id" \
    --k100-batch-id "$k100_batch_id" \
    --output-csv "$selection_csv" \
    --config-dir "$config_dir"

profile_submit_output="$(
    slurm/submit_cpu_search_v3_profile_selected.sh \
        --selection-csv "$selection_csv" \
        --batch-id "$profile_batch_id" \
        --dependency "afterok:${profile_build_job_id}"
)"
printf '%s\n' "$profile_submit_output"
profile_job_id="$(printf '%s\n' "$profile_submit_output" | awk -F= '/slurm_job_id=/{print $2}' | tail -n 1)"
[[ -n "$profile_job_id" ]] || die "failed to capture profile job id"

finalize_wrap="$(printf '%q ' \
    python3 \
    "${repo_root}/script/parse_cpu_search_v3_profile_logs.py" \
    --batch-id "${profile_batch_id}" \
    --repo-root "${repo_root}" \
    "${profile_log_dir}")"
finalize_wrap+=$'\n'
finalize_wrap+="$(printf '%q ' \
    python3 \
    "${repo_root}/script/plot_cpu_search_v3_stage_breakdown.py" \
    --profile-csv "${repo_root}/profiling/cpu_search_v3_profile/profile_summary_${profile_batch_id}.csv" \
    --output-prefix "${repo_root}/profiling/cpu_search_v3_profile/cpu_search_v3_stage_breakdown_${profile_batch_id}")"

finalize_job_id="$(
    sbatch \
        --parsable \
        --chdir "$repo_root" \
        --job-name "cpu_search_v3-finalize-${profile_batch_id}" \
        --dependency "afterok:${profile_job_id}" \
        --output "${repo_root}/slurm/logs/slurm_cpu_search_v3_finalize_${profile_batch_id}_%j.out" \
        --error "${repo_root}/slurm/logs/slurm_cpu_search_v3_finalize_${profile_batch_id}_%j.err" \
        --wrap "$finalize_wrap"
)"

echo "[followup] selection_csv=${selection_csv}"
echo "[followup] profile_job_id=${profile_job_id}"
echo "[followup] finalize_job_id=${finalize_job_id}"
