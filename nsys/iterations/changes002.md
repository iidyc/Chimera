# Iteration 002

## Goal

Create a focused Nsight Systems report that starts immediately before the
first profiled build-time CAGRA centroid-assignment search and stops after the
Nth profiled CAGRA search completes.

## Code changes

- Added NVTX push/pop markers for the focused CAGRA-search profiling window.
- Used CUDA profiler API start/stop as the actual Nsight capture gate after
  `--capture-range=nvtx` failed to produce reports for the emitted NVTX ranges
  with this Nsight Systems build.
- Added configurable iteration names through
  `CHIMERA_BUILD_ASSIGN_PROFILE_ITERATION`.
- For iteration `iterXXX`, the focused capture uses:
  - range start: `iterXXX_start`
  - end marker: `iterXXX_end`
- The D2H label copy for profiled batches is delayed until after the CAGRA
  search has completed and the focused NVTX range has ended, so the report is
  centered on CAGRA search rather than assignment cleanup.

## Run command shape

```bash
CHIMERA_BUILD_ASSIGN_PROFILE_BATCHES=3 \
CHIMERA_BUILD_ASSIGN_PROFILE_ITERATION=iter002 \
CUDA_VISIBLE_DEVICES=0 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat-shutdown:10 \
  --sample=none \
  --force-overwrite=true \
  -o nsys/iterations/iter002 \
  conda run -n chimera Chimera/build/gpu_build ...
```

## Expected artifact

- `nsys/iterations/iter002.nsys-rep`

## Result

Generated:

- `nsys/iterations/iter002.nsys-rep`

Report size:

- `200 KiB`

Focused report summary:

- NVTX range: `iter002_start`
- NVTX end marker: `iter002_end`
- CUDA GPU kernels in report: `7`
- CAGRA search kernels: `6`
- Tiny dataset descriptor init kernels: `1`

This removed unrelated CAGRA graph construction, quantization, and later build
work from the report.
