# Iteration 003

## Goal

Measure build-time assignment behavior for the real 1M-centroid case instead
of the earlier 4K-centroid synthetic run.

## Profiling setup

- Dataset: `dataset/lotte/raw/data.bin`
- Centroids/clusters: `1000000`
- Assignment batch size: `65536`
- Profiled assignment batches: `10`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Nsight output: `nsys/iterations/iter003.nsys-rep`

## Code changes

- Added fixed NVTX markers around the CAGRA search launch call:
  - `build_index:carga_search_start`
  - `build_index:carga_search_end`
- Kept CUDA profiler API start/stop as the reliable capture gate.
- Added profiling-only early exit:
  - `CHIMERA_BUILD_ASSIGN_PROFILE_EXIT_AFTER_CAPTURE=1`
  - This exits after the focused assignment capture so the run does not
    continue through full LoTTE assignment and quantization.

## Command shape

```bash
CUDA_VISIBLE_DEVICES=0 \
CHIMERA_BUILD_ASSIGN_PROFILE_BATCHES=10 \
CHIMERA_BUILD_ASSIGN_PROFILE_EXIT_AFTER_CAPTURE=1 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat-shutdown:10 \
  --sample=none \
  --force-overwrite=true \
  -o nsys/iterations/iter003 \
  Chimera/build/gpu_build \
    --index_dir /tmp/chimera_lotte_1m_assign_profile/index \
    --doclens dataset/lotte/raw/doclens.bin \
    --data dataset/lotte/raw/data.bin \
    --n_clusters 1000000
```
