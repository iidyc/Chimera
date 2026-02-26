#pragma once

// This file previously contained GPU-side heap management and token-distance
// extraction kernels for the overlapping Stage 2+3 pipeline.
//
// The new overlap architecture no longer needs GPU-side kernels:
//   - GPU runs only binary IP + doc_score (from gpu_kernels.cuh)
//   - D2H copies transfer results to CPU via cudaMemcpy2DAsync
//   - CPU does all refinement and top-k selection
//
// This file is kept as an empty placeholder to avoid breaking includes.
