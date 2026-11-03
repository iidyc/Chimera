#!/usr/bin/env bash
set -euo pipefail
root=/scratch3/workspace/juelinliu_umass_edu-gpu-mvr/GIGP-plus
export PATH=/home/juelinliu_umass_edu/conda/envs/IGP/bin:$PATH
export LD_LIBRARY_PATH="$root/runtime:/home/juelinliu_umass_edu/conda/envs/IGP/lib:${LD_LIBRARY_PATH:-}"
exec python "$root/benchmark/prepare_inputs.py" \
  --dataset-root /scratch3/workspace/juelinliu_umass_edu-gpu-mvr/Chimera/dataset \
  --output "$root/data" --dataset "$1" --bits "${2:-4}" \
  --prepare-binary "$root/bin/prepare_ivf"
