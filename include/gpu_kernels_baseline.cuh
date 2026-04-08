#pragma once

// Baseline (unoptimized) GPU kernels for performance comparison.
// All kernels produce identical results to the optimized versions in gpu_kernels.cuh
// but use the most naive, straightforward implementations:
//   - No shared memory for query vectors or LUTs
//   - No vectorized loads (byte-by-byte binary code access)
//   - No __ffsll or other bit-manipulation intrinsics
//   - No tiling or cooperative loading patterns
//   - No block-level reductions in doc scoring
//   - No sparse document tracking

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include "gpu_config.cuh"

// ============================================================
// Stage 1: Naive binary inner product kernel
// ============================================================
// One thread per (query, embedding) pair.
// No shared memory — each thread reads its query vector from global memory.
// No vectorized code loads — reads bytes individually.
// No __ffsll — simple for-loop over all PADDED_DIM bits.

__global__ void stage1_binary_ip_baseline_kernel(
    const float* __restrict__ d_queries,        // [Q_DOCLEN * PADDED_DIM]
    const char*  __restrict__ d_one_bit_code,   // [n * CODE_BYTES]
    const float* __restrict__ d_one_bit_factor, // [n]
    const float* __restrict__ d_cb1_sumq,       // [Q_DOCLEN]
    const size_t* __restrict__ d_emb_ids,       // [total_pairs]
    const int*   __restrict__ d_pair_offsets,    // [Q_DOCLEN + 1]
    float* __restrict__ d_out_dists,            // [Q_DOCLEN * max_embs_per_query]
    size_t max_embs_per_query
) {
    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) return;

    const float cb1_sumq = d_cb1_sumq[query_idx];
    const size_t pair_start = d_pair_offsets[query_idx];
    const size_t pair_end   = d_pair_offsets[query_idx + 1];
    const size_t num_embs   = pair_end - pair_start;

    // Grid-stride loop over embeddings
    for (size_t idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         idx < num_embs;
         idx += (size_t)blockDim.x * gridDim.x)
    {
        const size_t emb_id = d_emb_ids[pair_start + idx];
        const char* code = d_one_bit_code + emb_id * CODE_BYTES;

        // Naive bit-by-bit inner product: read each byte, check each bit
        float ip = 0.0f;
        for (int byte_idx = 0; byte_idx < CODE_BYTES; byte_idx++) {
            unsigned char byte_val = (unsigned char)code[byte_idx];
            for (int bit = 0; bit < 8; bit++) {
                if (byte_val & (1u << bit)) {
                    int dim = byte_idx * 8 + bit;
                    ip += d_queries[query_idx * PADDED_DIM + dim];
                }
            }
        }

        float dist = (ip - cb1_sumq) * d_one_bit_factor[emb_id];
        d_out_dists[query_idx * max_embs_per_query + idx] = dist;
    }
}

// ============================================================
// Stage 2: Naive binary inner product kernel
// ============================================================
// 2D grid: blockIdx.x covers tokens, blockIdx.y covers queries.
// Each thread handles one (token, query) pair independently.
// No shared memory — reads query and code from global memory each time.

__global__ void stage2_binary_ip_baseline_kernel(
    const float* __restrict__ d_queries,        // [Q_DOCLEN * PADDED_DIM]
    const char*  __restrict__ d_one_bit_code,   // [n * CODE_BYTES]
    const float* __restrict__ d_one_bit_factor, // [n]
    const float* __restrict__ d_cb1_sumq,       // [Q_DOCLEN]
    const size_t* __restrict__ d_token_ids,     // [total_tokens]
    float* __restrict__ d_out_dists,            // [Q_DOCLEN * total_tokens]
    size_t total_tokens,
    size_t batch_tokens
) {
    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) return;

    const float cb1_sumq = d_cb1_sumq[query_idx];

    // Grid-stride loop over tokens
    for (size_t tok_idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         tok_idx < batch_tokens;
         tok_idx += (size_t)blockDim.x * gridDim.x)
    {
        const size_t token_id = d_token_ids[tok_idx];
        const float factor = d_one_bit_factor[token_id];
        const char* code = d_one_bit_code + token_id * CODE_BYTES;

        // Naive bit-by-bit inner product
        float ip = 0.0f;
        for (int byte_idx = 0; byte_idx < CODE_BYTES; byte_idx++) {
            unsigned char byte_val = (unsigned char)code[byte_idx];
            for (int bit = 0; bit < 8; bit++) {
                if (byte_val & (1u << bit)) {
                    int dim = byte_idx * 8 + bit;
                    ip += d_queries[query_idx * PADDED_DIM + dim];
                }
            }
        }

        d_out_dists[query_idx * total_tokens + tok_idx] =
            (ip - cb1_sumq) * factor;
    }
}

// ============================================================
// Naive document scoring kernel
// ============================================================
// One thread per candidate document. No shared memory, no tiling,
// no block-level reduction. Each thread loops over all queries and
// all tokens for its assigned document.

__global__ void doc_score_baseline_kernel(
    const float*  __restrict__ d_token_dists,       // [Q_DOCLEN][total_tokens]
    const size_t* __restrict__ d_candidate_offsets,  // [num_candidates + 1]
    float*        d_doc_scores,                      // [num_candidates]
    size_t total_tokens,
    size_t num_candidates
) {
    size_t cand_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand_idx >= num_candidates) return;

    size_t tok_start = d_candidate_offsets[cand_idx];
    size_t tok_end   = d_candidate_offsets[cand_idx + 1];

    float score = 0.0f;
    for (int q = 0; q < Q_DOCLEN; q++) {
        float max_val = -FLT_MAX;
        for (size_t t = tok_start; t < tok_end; t++) {
            float val = d_token_dists[q * total_tokens + t];
            if (val > max_val) max_val = val;
        }
        score += max_val;
    }

    d_doc_scores[cand_idx] = score;
}

// ============================================================
// Naive extract kernel
// ============================================================
// One thread per output element. No coalescing optimization.
// Reads from [query][token] layout, writes to [token][query] layout.

__global__ void extract_one_bit_dists_baseline_kernel(
    const float*  __restrict__ d_token_dists,       // [Q_DOCLEN][total_tokens]
    const size_t* __restrict__ d_candidate_offsets,
    const int*    __restrict__ d_selected_indices,
    float*        d_out_one_bit_dists,               // [token][query] output layout
    const size_t* __restrict__ d_out_offsets,
    size_t total_tokens,
    size_t k
) {
    // Global thread index across all elements of all selected candidates
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // We need to know total elements to bound the thread. Since we don't have it
    // as a parameter, iterate over selected candidates and find ours.
    // Simple approach: each block handles one selected candidate (like the optimized version)
    size_t sel_idx = blockIdx.x;
    if (sel_idx >= k) return;

    size_t cand_idx = d_selected_indices[sel_idx];
    size_t tok_start = d_candidate_offsets[cand_idx];
    size_t tok_end   = d_candidate_offsets[cand_idx + 1];
    size_t num_tokens = tok_end - tok_start;
    size_t out_base = d_out_offsets[sel_idx];

    // Each thread handles elements within this candidate
    for (size_t i = threadIdx.x; i < num_tokens * Q_DOCLEN; i += blockDim.x) {
        size_t t_local = i / Q_DOCLEN;
        size_t q_idx   = i % Q_DOCLEN;

        float val = d_token_dists[q_idx * total_tokens + tok_start + t_local];
        d_out_one_bit_dists[(out_base + t_local) * Q_DOCLEN + q_idx] = val;
    }
}

// ============================================================
// Naive Stage 1 aggregation kernel (no sparse doc tracking)
// ============================================================
// Same atomic max-pool as the optimized version, but without the
// touched-doc tracking machinery. Relies on full-matrix memset
// instead of sparse cleanup.

// Atomic max for floats using compare-and-swap (same as optimized)
__device__ __forceinline__ float atomicMaxFloat_baseline(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                       __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void aggregate_stage1_baseline_kernel(
    const size_t* __restrict__ d_emb_ids,
    const float*  __restrict__ d_emb_dists,       // [Q_DOCLEN * max_embs_per_query]
    const int*    __restrict__ d_pair_offsets,     // [Q_DOCLEN + 1]
    const int*    __restrict__ d_doc_ids,          // [n]
    float*        d_doc_query_max,                 // [Q_DOCLEN * num_docs] transposed
    size_t        num_docs,
    size_t        total_pairs,
    size_t        max_embs_per_query
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    // Linear search for query index (naive, no binary search)
    int q_idx = 0;
    for (int q = 0; q < Q_DOCLEN; q++) {
        if (idx >= (size_t)d_pair_offsets[q + 1]) {
            q_idx = q + 1;
        }
    }

    size_t emb_id = d_emb_ids[idx];
    int doc_id = d_doc_ids[emb_id];

    int local_pair_idx = (int)(idx - (size_t)d_pair_offsets[q_idx]);
    float dist = d_emb_dists[(size_t)q_idx * max_embs_per_query + local_pair_idx];

    if (doc_id >= (int)num_docs) return;

    // Atomic max into transposed matrix [q_idx * num_docs + doc_id]
    size_t matrix_idx = (size_t)q_idx * num_docs + doc_id;
    atomicMaxFloat_baseline(&d_doc_query_max[matrix_idx], dist);
}

// ============================================================
// Naive sum doc scores kernel (iterates ALL documents)
// ============================================================
// One thread per document. Sums across Q_DOCLEN queries.
// No sparse tracking — runs over the entire document set.

__global__ void sum_doc_scores_baseline_kernel(
    const float* __restrict__ d_doc_query_max,  // [Q_DOCLEN * num_docs] transposed
    float*       d_doc_scores,                   // [num_docs]
    int*         d_doc_ids_out,                  // [num_docs] — writes doc_id = thread index
    size_t       num_docs
) {
    size_t doc_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (doc_id >= num_docs) return;

    float score = 0.0f;
    for (int q = 0; q < Q_DOCLEN; q++) {
        score += d_doc_query_max[(size_t)q * num_docs + doc_id];
    }
    d_doc_scores[doc_id] = score;
    d_doc_ids_out[doc_id] = (int)doc_id;
}

// ============================================================
// Utility kernels — reused from optimized version (identical)
// ============================================================
// These are not compute bottlenecks and are already straightforward.
// They are included here so gpu_index_baseline.cuh is self-contained.

__global__ void gather_token_ids_baseline_kernel(
    const int*    __restrict__ d_candidate_doc_ids,
    const int*    __restrict__ d_doc_ptrs,
    const size_t* __restrict__ d_candidate_offsets,
    size_t*       d_out_token_ids,
    size_t num_candidates
) {
    size_t cand_idx = blockIdx.x;
    if (cand_idx >= num_candidates) return;

    int doc_id = d_candidate_doc_ids[cand_idx];
    int doc_start = d_doc_ptrs[doc_id];
    int doc_end = d_doc_ptrs[doc_id + 1];
    size_t out_offset = d_candidate_offsets[cand_idx];

    for (int t = threadIdx.x; t < (doc_end - doc_start); t += blockDim.x) {
        d_out_token_ids[out_offset + t] = doc_start + t;
    }
}

__global__ void compute_query_expansion_sizes_baseline_kernel(
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*       __restrict__ d_cluster_pos,
    int*                d_query_sizes,
    int                 nprobe,
    size_t              n_clusters,
    size_t              num_queries
) {
    size_t query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;

    int total_size = 0;
    for (int p = 0; p < nprobe; ++p) {
        uint32_t cluster_id = d_cagra_labels[query_idx * nprobe + p];
        if (cluster_id >= (uint32_t)n_clusters) continue;
        size_t cluster_size = d_cluster_pos[cluster_id + 1] - d_cluster_pos[cluster_id];
        total_size += (int)cluster_size;
    }
    d_query_sizes[query_idx] = total_size;
}

__global__ void expand_cluster_ids_baseline_kernel(
    const uint32_t* __restrict__ d_cagra_labels,
    const int*          __restrict__ d_inv_list,
    const size_t*       __restrict__ d_cluster_pos,
    const int*          __restrict__ d_query_offsets,
    size_t*             d_emb_ids,
    int                 nprobe,
    size_t              n_clusters,
    size_t              num_queries
) {
    size_t query_idx = blockIdx.x;
    if (query_idx >= num_queries) return;

    size_t out_start = d_query_offsets[query_idx];
    size_t write_pos = 0;

    for (int p = 0; p < nprobe; ++p) {
        uint32_t cluster_id = d_cagra_labels[query_idx * nprobe + p];
        if (cluster_id >= (uint32_t)n_clusters) continue;

        size_t cluster_start = d_cluster_pos[cluster_id];
        size_t cluster_end = d_cluster_pos[cluster_id + 1];
        size_t cluster_size = cluster_end - cluster_start;

        for (size_t i = threadIdx.x; i < cluster_size; i += blockDim.x) {
            d_emb_ids[out_start + write_pos + i] = d_inv_list[cluster_start + i];
        }

        write_pos += cluster_size;
        __syncthreads();
    }
}

__global__ void gather_doc_lengths_baseline_kernel(
    const int* __restrict__ d_topk_doc_ids,
    const int* __restrict__ d_doc_ptrs,
    int*       d_doc_lengths,
    size_t k
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;

    int doc_id = d_topk_doc_ids[idx];
    d_doc_lengths[idx] = d_doc_ptrs[doc_id + 1] - d_doc_ptrs[doc_id];
}
