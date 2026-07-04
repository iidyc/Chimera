# Iteration 001

## Goal

Profile the build-time CAGRA centroid-assignment loop before deciding whether
to add a static CAGRA workspace pool or CUDA Graph capture.

## Code changes

- Split `gpu_build` helper logic out of `cpp/cuda/src/build_index.cu` into
  build-index-specific CUDA modules:
  - `cpp/cuda/src/build_index/assignment.cu`
  - `cpp/cuda/src/build_index/build_utils.cu`
  - `cpp/cuda/src/build_index/centroid_sampling.cu`
- Added matching headers under `cpp/cuda/include/build_index/`.
- Added block-based centroid sampling defaults:
  - `bucket_size = 256`
  - `seed = 41`
- Added metadata fields for the actual centroid sampling seed and bucket size.
- Added an opt-in assignment profiler controlled by
  `CHIMERA_BUILD_ASSIGN_PROFILE_BATCHES`.

## Profiling scope

The profiler reports the first N assignment batches with:

- CPU mmap-to-pinned copy time.
- CPU enqueue time for H2D, CAGRA search, D2H, and event records.
- GPU H2D time.
- GPU CAGRA search time.
- GPU D2H time.
- Total stream time from H2D start to D2H completion.

## Expected artifact

- `nsys/iterations/iter001.nsys-prof`

## Run

Command shape:

```bash
CHIMERA_BUILD_ASSIGN_PROFILE_BATCHES=3 nsys profile ... gpu_build \
  --index_dir /tmp/chimera_assign_profile/index \
  --doclens /tmp/chimera_assign_profile/raw/doclens.bin \
  --data /tmp/chimera_assign_profile/raw/data.bin \
  --n_clusters 4096
```

Synthetic dataset:

- `n = 196608`
- `d = 128`
- `n_clusters = 4096`
- assignment batch size: `65536`
- visible GPUs used by assignment: `3`

## Results

| batch | device | start | count | host_load_ms | host_enqueue_ms | gpu_h2d_ms | gpu_cagra_ms | gpu_d2h_ms | gpu_total_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 0 | 65536 | 12.987 | 289.729 | 2.790 | 286.872 | 0.053 | 289.715 |
| 2 | 1 | 131072 | 65536 | 11.017 | 332.415 | 2.759 | 329.603 | 0.059 | 332.421 |
| 1 | 0 | 65536 | 13.155 | 360.801 | 3.005 | 357.743 | 0.047 | 360.794 |

Average over the three profiled batches:

- CPU mmap-to-pinned copy: `12.386 ms`
- GPU H2D: `2.851 ms`
- GPU CAGRA search: `324.739 ms`
- GPU D2H: `0.053 ms`
- GPU total stream time: `327.643 ms`

## First-pass conclusion

The bottleneck is the CAGRA centroid search itself. H2D and D2H are small
relative to CAGRA, and CPU mmap-to-pinned loading is also much smaller than the
search time. A static RAFT/CAGRA workspace pool may still reduce allocator
variance, but CUDA Graph capture is unlikely to be the main win unless nsys
shows significant launch/allocator gaps inside the CAGRA region.

## Kernel launches per assignment batch

Each profiled assignment batch launched `3` kernels on its worker stream:

- `1` tiny `standard_dataset_descriptor_init_kernel`, about `0.002 ms`.
- `2` `single_cta_search::search_kernel` CAGRA kernels.

The two CAGRA kernels were roughly:

- main search kernel: `147.9-149.2 ms`
- short follow-up search kernel: `2.0 ms`

So if counting only CAGRA search work, the assignment path launches `2` kernels
per batch. If counting all kernels in the per-batch search call region, it is
`3` kernels per batch.
