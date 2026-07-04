# Iteration 004

## Goal

Double the build-assignment batch size and check whether 1M-centroid CAGRA
assignment time scales roughly linearly with batch size.

## Profiling setup

- Dataset: `dataset/lotte/raw/data.bin`
- Centroids/clusters: `1000000`
- Assignment batch size: `131072`
- Profiled assignment batches: `10`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Nsight output: `nsys/iterations/iter004.nsys-rep`

## Code changes

- Added `CHIMERA_BUILD_ASSIGN_BATCH_SIZE` as an environment override for the
  build-time assignment batch size.
- Default remains `65536`.

## Command shape

```bash
CUDA_VISIBLE_DEVICES=0 \
CHIMERA_BUILD_ASSIGN_BATCH_SIZE=131072 \
CHIMERA_BUILD_ASSIGN_PROFILE_BATCHES=10 \
CHIMERA_BUILD_ASSIGN_PROFILE_EXIT_AFTER_CAPTURE=1 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat-shutdown:10 \
  --sample=none \
  --force-overwrite=true \
  -o nsys/iterations/iter004 \
  Chimera/build/gpu_build \
    --index_dir /tmp/chimera_lotte_1m_assign_profile_b131072/index \
    --doclens dataset/lotte/raw/doclens.bin \
    --data dataset/lotte/raw/data.bin \
    --n_clusters 1000000
```

## Results

Per-batch CAGRA time from the Nsight kernel timeline:

| batch | CAGRA kernels | CAGRA time ms |
| ---: | ---: | ---: |
| 0 | 3 | 518.519 |
| 1 | 3 | 526.387 |
| 2 | 3 | 527.486 |
| 3 | 3 | 527.667 |
| 4 | 3 | 527.823 |
| 5 | 3 | 527.714 |
| 6 | 3 | 527.696 |
| 7 | 3 | 527.909 |
| 8 | 3 | 526.675 |
| 9 | 3 | 530.388 |

Steady-state average excluding batch 0:

- `527.749 ms` per `131072`-embedding batch.
- Throughput: `248360 embeddings/s/GPU`.

Comparison to `iter003` with `65536`-embedding batches and 1M centroids:

- `iter003` steady-state average excluding batch 0: `263.909 ms`.
- `iter004` steady-state average excluding batch 0: `527.749 ms`.
- Time ratio: `1.9997x`.
- Throughput is effectively unchanged.

Conclusion: doubling assignment batch size doubles per-batch CAGRA time almost
exactly. It does not improve per-GPU assignment throughput for the 1M-centroid
case.
