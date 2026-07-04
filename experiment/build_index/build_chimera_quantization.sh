#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  experiment/build_index/build_chimera_quantization.sh [options]

Build Chimera revision indices. By default this builds 4bit and 8bit 2M-centroid
indices for:
  lotte hotpot msmarco

Output names:
  dataset/<dataset>/chimera_2m_4bit
  dataset/<dataset>/chimera_2m_8bit

Options:
  --dataset <name>   Dataset to build. Repeatable. Default: lotte hotpot msmarco.
  --bits <mode>      4bit, 8bit, or both. Default: both.
  --dry-run          Print commands and metadata without running gpu_build.
  -h, --help         Show this help message.

The script expects the Chimera runtime environment to already be active on the
target host. It writes raw build logs under:
  revision_experiments_gpu/build_time/logs/<dataset>/<index_name>/
EOF
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

dry_run=0
bits="both"
datasets=()
output_root="${CHIMERA_REVISION_OUTPUT_ROOT:-revision_experiments_gpu}"
gpu_model="${CHIMERA_REVISION_GPU_MODEL:-unknown}"
n_clusters=2000000
n_clusters_label="2m"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      [[ $# -ge 2 ]] || { echo "error: missing value for --dataset" >&2; exit 1; }
      datasets+=("$2")
      shift 2
      ;;
    --bits)
      [[ $# -ge 2 ]] || { echo "error: missing value for --bits" >&2; exit 1; }
      bits="$2"
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
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#datasets[@]} -eq 0 ]]; then
  datasets=(lotte hotpot msmarco)
fi

case "$bits" in
  4bit)
    bit_modes=(4bit)
    ;;
  8bit)
    bit_modes=(8bit)
    ;;
  both)
    bit_modes=(4bit 8bit)
    ;;
  *)
    echo "error: --bits must be 4bit, 8bit, or both" >&2
    exit 1
    ;;
esac

validate_dataset_name() {
  case "$1" in
    ""|.|..|*/*)
      echo "error: dataset name must be a simple directory name: $1" >&2
      exit 1
      ;;
  esac
}

gpu_build_bin="${repo_root}/Chimera/build/gpu_build"
if [[ ! -x "$gpu_build_bin" && $dry_run -eq 0 ]]; then
  echo "error: missing executable ${gpu_build_bin}" >&2
  exit 1
fi

make_run_id() {
  printf '%s_pid%s' "$(date -u +%Y%m%dT%H%M%SZ)" "$$"
}

cpu_model() {
  if command -v lscpu >/dev/null 2>&1; then
    lscpu | awk -F: '/Model name:/ {gsub(/^[ \t]+/, "", $2); print $2; exit}'
  elif [[ -r /proc/cpuinfo ]]; then
    awk -F: '/model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo
  else
    echo unknown
  fi
}

gpu_count_used() {
  if [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
    local cleaned="${CUDA_VISIBLE_DEVICES// /}"
    local count=0
    local gpu_id
    IFS=',' read -r -a gpu_ids <<< "$cleaned"
    for gpu_id in "${gpu_ids[@]}"; do
      [[ -n "$gpu_id" ]] && ((count += 1))
    done
    echo "$count"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l | tr -d '[:space:]'
  else
    echo unknown
  fi
}

log_line() {
  local log_file="$1"
  local line="$2"
  if [[ $dry_run -eq 1 ]]; then
    echo "$line"
  else
    echo "$line" | tee -a "$log_file"
  fi
}

for dataset in "${datasets[@]}"; do
  validate_dataset_name "$dataset"
  raw_dir="${repo_root}/dataset/${dataset}/raw"
  data_file="${raw_dir}/data.bin"
  doclens_file="${raw_dir}/doclens.bin"

  if [[ $dry_run -eq 0 ]]; then
    [[ -f "$data_file" ]] || { echo "error: missing ${data_file}" >&2; exit 1; }
    [[ -f "$doclens_file" ]] || { echo "error: missing ${doclens_file}" >&2; exit 1; }
  elif [[ ! -f "$data_file" || ! -f "$doclens_file" ]]; then
    echo "[dry-run] warning: raw inputs not present for dataset=${dataset}; command is for target host."
  fi

  for bit_mode in "${bit_modes[@]}"; do
    index_name="chimera_${n_clusters_label}_${bit_mode}"
    index_dir="${repo_root}/dataset/${dataset}/${index_name}"
    log_dir="${repo_root}/${output_root}/build_time/logs/${dataset}/${index_name}"
    run_id="$(make_run_id)"
    log_file="${log_dir}/gpu_build_${run_id}.log"
    cmd=(
      "$gpu_build_bin"
      --index_dir "$index_dir"
      --doclens "$doclens_file"
      --data "$data_file"
      --n_clusters "$n_clusters"
      --centroid_sample_bucket_size 256
      --centroid_sample_seed 41
      --build_mode gpu-search-minimal
      --quantization "$bit_mode"
      --quantization_backend gpu-merged
    )

    if [[ $dry_run -eq 0 ]]; then
      mkdir -p "$index_dir" "$log_dir"
      : > "$log_file"
    fi

    log_line "$log_file" "[driver] repo_root=${repo_root}"
    log_line "$log_file" "[driver] dataset=${dataset}"
    log_line "$log_file" "[driver] gpu_model=${gpu_model}"
    log_line "$log_file" "[driver] data_file=${data_file}"
    log_line "$log_file" "[driver] doclens_file=${doclens_file}"
    log_line "$log_file" "[driver] index_dir=${index_dir}"
    log_line "$log_file" "[driver] run_id=${run_id}"
    log_line "$log_file" "[driver] log_file=${log_file}"
    log_line "$log_file" "[driver] n_clusters=${n_clusters}"
    log_line "$log_file" "[driver] centroid_sample_bucket_size=256"
    log_line "$log_file" "[driver] centroid_sample_seed=41"
    log_line "$log_file" "[driver] build_mode=gpu-search-minimal"
    log_line "$log_file" "[driver] quantization=${bit_mode}"
    log_line "$log_file" "[driver] quantization_backend=gpu-merged"
    log_line "$log_file" "[driver] cpu_model=$(cpu_model)"
    log_line "$log_file" "[driver] cpu_threads_available=$(nproc)"
    log_line "$log_file" "[driver] cpu_threads_used=${OMP_NUM_THREADS:-$(nproc)}"
    log_line "$log_file" "[driver] omp_num_threads=${OMP_NUM_THREADS-<unset>}"
    log_line "$log_file" "[driver] cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
    log_line "$log_file" "[driver] gpu_count_used=$(gpu_count_used)"
    log_line "$log_file" "[driver] command=$(printf '%q ' "${cmd[@]}")"

    if [[ $dry_run -eq 1 ]]; then
      echo "[dry-run] $(printf '%q ' "${cmd[@]}")"
    else
      if command -v /usr/bin/time >/dev/null 2>&1; then
        /usr/bin/time -v "${cmd[@]}" 2>&1 | tee -a "$log_file"
      else
        "${cmd[@]}" 2>&1 | tee -a "$log_file"
      fi
    fi
  done
done
