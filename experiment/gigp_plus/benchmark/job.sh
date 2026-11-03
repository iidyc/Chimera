#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
dataset="$2"
mode="${3:-smoke}"
bits="${4:-4}"
[[ "$bits" == 4 || "$bits" == 2 ]]
root=/scratch3/workspace/juelinliu_umass_edu-gpu-mvr/GIGP-plus
export PATH=/home/juelinliu_umass_edu/conda/envs/IGP/bin:$PATH
export LD_LIBRARY_PATH="/home/juelinliu_umass_edu/conda/envs/IGP/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=3 OMP_PROC_BIND=close OMP_PLACES=cores
export OMP_DISPLAY_AFFINITY=true
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
topology="$(python "$root/benchmark/hardware_probe.py" --platform "$gpu")"
mapfile -t hardware <<< "$topology"
gpu_model="${hardware[0]}"
cpu_model="${hardware[1]}"
node="${hardware[2]}"
allowed="${hardware[3]}"
data="$root/data"
output="${GIGP_OUTPUT_ROOT:-$root/results/${gpu}_${dataset}_4bit/$mode}"
if [[ "$bits" == 2 ]]; then
  data="$root/data_paper2bit"
  output="${GIGP_OUTPUT_ROOT:-$root/paper2bit_results/${gpu}_${dataset}_2bit/$mode}"
fi
mkdir -p "$output"
bash "$root/benchmark/build_worker.sh" > "$output/build_${SLURM_JOB_ID}.log" 2>&1
binary="/tmp/gigpp_build_${SLURM_JOB_ID}/IGPGPUPP"
{
  date -u
  echo "gpu=$gpu_model cpu=$cpu_model numa_node=$node allowed_cpus=$allowed"
  echo "workers=3 OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "cuda_device_pci=${hardware[4]} cuda_device_uuid=${hardware[5]}"
  nvidia-smi
  sha256sum "$binary"
  ldd "$binary"
} > "$output/hardware_${SLURM_JOB_ID}.txt"
args=()
[[ "$mode" != smoke ]] || args+=(--smoke)
[[ "$mode" != pilot ]] || args+=(--pilot)
[[ "$bits" != 2 ]] || args+=(--paper-default)
exec numactl --physcpubind="$allowed" --membind="$node" \
  python "$root/benchmark/sweep.py" --binary "$binary" \
  --data "$data" --output "$output" --dataset "$dataset" --gpu "$gpu" --bits "$bits" "${args[@]}"
