#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
dataset="$2"
mode="${3:-pilot}"
root=/scratch3/workspace/juelinliu_umass_edu-gpu-mvr/GIGP-plus
source_root="$root/paired/source"
env_root=/home/juelinliu_umass_edu/conda/envs/gpu-mvr
export PATH="$env_root/bin:$PATH"
export LD_LIBRARY_PATH="$env_root/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}" OMP_PROC_BIND=false OMP_PLACES=cores
export OMP_DISPLAY_AFFINITY=true PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
topology="$(python "$root/benchmark/hardware_probe.py" --platform "$gpu")"
mapfile -t hardware <<< "$topology"
gpu_model="${hardware[0]}"
cpu_model="${hardware[1]}"
node="${hardware[2]}"
allowed="${hardware[3]}"
export CHIMERA_SLOT_CPUSETS="$(python - <<'PY'
import os
cpus = sorted(os.sched_getaffinity(0))
assert len(cpus) >= 3
groups = [cpus[i * len(cpus) // 3:(i + 1) * len(cpus) // 3] for i in range(3)]
print(';'.join(','.join(map(str, group)) for group in groups))
PY
)"
stage3_threads="$((SLURM_CPUS_PER_TASK / 3))"
output="$root/results/chimera3_${gpu}_${dataset}_4bit/$mode"
mkdir -p "$output"
build="/tmp/chimera3_build_${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}"
[[ ! -e "$build" ]]
{
  cmake -S "$source_root/Chimera" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCHIMERA_ARCH_FLAGS=-march=native -DCHIMERA_ENABLE_PROFILE=OFF -DCHIMERA_USE_CCACHE=OFF
  cmake --build "$build" --target gpu_search --parallel 4
} > "$output/build_${SLURM_JOB_ID}.log" 2>&1
{
  date -u
  echo "gpu=$gpu_model cpu=$cpu_model numa=$node allowed_cpus=$allowed slots=3"
  echo "slot_cpu_groups=$CHIMERA_SLOT_CPUSETS stage3_threads_per_slot=$stage3_threads"
  echo "cuda_device_pci=${hardware[4]} cuda_device_uuid=${hardware[5]}"
  sha256sum "$root/paired/source_pinned_slots3.tar" "$build/gpu_search"
  nvidia-smi
  ldd "$build/gpu_search"
} > "$output/hardware_${SLURM_JOB_ID}.txt"
args=()
if [[ "$mode" == pilot ]]; then
  case "$dataset" in lotte) args=(--config-labels c1 f1);; hotpot) args=(--config-labels g1 k1);; msmarco) args=(--config-labels b4 e1);; esac
fi
cd "$source_root"
exec numactl --physcpubind="$allowed" --membind="$node" python \
  experiment/chimera/run_gpu_search_topk_baseline.py --repo-root "$source_root" \
  --build-dir "$build" --datasets "$dataset" --slots 3 --num-runs 3 --k 100 \
  --nq -1 --warmup 5 --stage3-threads "$stage3_threads" --omp-num-threads "$OMP_NUM_THREADS" \
  --index-name chimera_2m_4bit_chimera_v35 --output-root "$output" \
  --run-id "job${SLURM_JOB_ID}" "${args[@]}"
