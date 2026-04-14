#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <stdexcept>

#include <cuda_runtime.h>

#include "gpu_kernels_v2.cuh"

__global__ void stage1_binary_ip_kernel_v2(
    const float* __restrict__ d_queries,
    const char*  __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_emb_ids,
    const int*   __restrict__ d_pair_offsets,
    float* __restrict__ d_out_dists,
    size_t max_embs_per_query
) {
    __shared__ float smem_query[PADDED_DIM];  // 512 bytes

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) return;

    // Coalesced load of query vector into shared memory
    const float* q_ptr = d_queries + query_idx * PADDED_DIM;
    #pragma unroll
    for (int i = threadIdx.x; i < PADDED_DIM; i += blockDim.x) {
        smem_query[i] = q_ptr[i];
    }
    __syncthreads();

    const float cb1_sumq = d_cb1_sumq[query_idx];
    const size_t pair_start = d_pair_offsets[query_idx];
    const size_t pair_end   = d_pair_offsets[query_idx + 1];
    const size_t num_embs   = pair_end - pair_start;

    // Grid-stride loop: each thread handles one embedding per iteration
    for (size_t idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         idx < num_embs;
         idx += (size_t)blockDim.x * gridDim.x)
    {
        const size_t emb_id = d_emb_ids[pair_start + idx];
        // Vectorized 128-bit load of binary code (2 x uint64_t)
        const uint64_t* code_ptr =
            (const uint64_t*)(d_one_bit_code + emb_id * CODE_BYTES);
        uint64_t code_regs[NUM_U64];
        code_regs[0] = code_ptr[0];
        code_regs[1] = code_ptr[1];

        float ip = 0.0f;
        #pragma unroll
        for (int blk = 0; blk < NUM_U64; blk++) {
            uint64_t bits = code_regs[blk];
            int base = blk * 64;
            // Skip zero bits using __ffsll intrinsic
            while (bits) {
                int pos = __ffsll(bits) - 1;
                ip += smem_query[base + pos];
                bits &= bits - 1;  // clear lowest set bit
            }
        }
        float dist = (ip - cb1_sumq) * d_one_bit_factor[emb_id];

        d_out_dists[query_idx * max_embs_per_query + idx] = dist;
    }
}

__global__ void stage2_binary_ip_kernel_v2(
    const float* __restrict__ d_queries,       // [Q_DOCLEN * PADDED_DIM]
    const char*  __restrict__ d_one_bit_code,  // [n * CODE_BYTES]
    const float* __restrict__ d_one_bit_factor,// [n]
    const float* __restrict__ d_cb1_sumq,      // [Q_DOCLEN]
    const size_t* __restrict__ d_token_ids,    // [total_tokens]
    float* __restrict__ d_out_dists,           // [Q_DOCLEN * total_tokens]
    size_t total_tokens,
    size_t batch_tokens
) {
    // Load ALL query vectors into shared memory (16 KB)
    __shared__ float smem_queries[Q_DOCLEN * PADDED_DIM];
    // Cache cb1_sumq in shared memory to avoid repeated global reads (128 B)
    __shared__ float smem_cb1_sumq[Q_DOCLEN];

    // Cooperative load: 256 threads load 32*128 = 4096 floats → 16 iterations
    const int total_query_floats = Q_DOCLEN * PADDED_DIM;
    for (int i = threadIdx.x; i < total_query_floats; i += blockDim.x) {
        smem_queries[i] = d_queries[i];
    }
    if (threadIdx.x < Q_DOCLEN) {
        smem_cb1_sumq[threadIdx.x] = d_cb1_sumq[threadIdx.x];
    }
    __syncthreads();

    // Grid-stride loop: each thread processes one token against all queries
    for (size_t tok_idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         tok_idx < batch_tokens;
         tok_idx += (size_t)blockDim.x * gridDim.x)
    {
        // Load binary code and factor ONCE per token
        const size_t token_id = d_token_ids[tok_idx];
        const float factor = d_one_bit_factor[token_id];

        // Load binary code into registers (2 x uint64_t)
        const uint64_t* code_ptr =
            (const uint64_t*)(d_one_bit_code + token_id * CODE_BYTES);
        uint64_t code_regs[NUM_U64];
        code_regs[0] = code_ptr[0];
        code_regs[1] = code_ptr[1];

        // Score against all Q_DOCLEN queries
        #pragma unroll
        for (int q = 0; q < Q_DOCLEN; q++) {
            const float* q_smem = smem_queries + q * PADDED_DIM;
            const float cb1_sumq = smem_cb1_sumq[q];

            float ip = 0.0f;
            #pragma unroll
            for (int blk = 0; blk < NUM_U64; blk++) {
                uint64_t bits = code_regs[blk];
                int base = blk * 64;
                // Fully unrolled loop — compiler emits predicated FADD with zero branching.
                // Do NOT replace with __ffsll: data-dependent branching causes warp divergence
                // multiplied 32x across queries, resulting in 3.6x slowdown.
                #pragma unroll
                for (int i = 0; i < 64; i++) {
                    if ((bits >> i) & 1ULL)
                        ip += q_smem[base + i];
                }
            }

            d_out_dists[q * total_tokens + tok_idx] =
                (ip - cb1_sumq) * factor;
        }
    }
}

// ============================================================
// LUT-based binary IP kernels
// ============================================================

__global__ void precompute_lut_kernel(
    const float* __restrict__ d_queries,  // [Q_DOCLEN * PADDED_DIM]
    float* __restrict__ d_lut             // [Q_DOCLEN * LUT_ENTRIES_PER_QUERY]
) {
    const int query_idx = blockIdx.x;
    if (query_idx >= Q_DOCLEN) return;

    const float* q_ptr = d_queries + query_idx * PADDED_DIM;
    float* lut_ptr = d_lut + query_idx * LUT_ENTRIES_PER_QUERY;

    // 512 entries total, 256 threads → 2 iterations
    for (int idx = threadIdx.x; idx < LUT_ENTRIES_PER_QUERY; idx += blockDim.x) {
        int chunk_idx = idx / LUT_SIZE;     // 0..31
        int lut_entry = idx % LUT_SIZE;     // 0..15 (the 4-bit pattern)

        int dim_start = chunk_idx * BITS_PER_CHUNK;
        float sum = 0.0f;

        #pragma unroll
        for (int bit_idx = 0; bit_idx < BITS_PER_CHUNK; bit_idx++) {
            if ((lut_entry >> bit_idx) & 1) {
                sum += q_ptr[dim_start + bit_idx];
            }
        }

        lut_ptr[idx] = sum;
    }
}

__global__ void stage1_binary_ip_lut_kernel(
    const float* __restrict__ d_lut,           // [Q_DOCLEN * LUT_ENTRIES_PER_QUERY]
    const char*  __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_emb_ids,
    const int*   __restrict__ d_pair_offsets,
    float* __restrict__ d_out_dists,
    size_t max_embs_per_query
) {
    __shared__ float smem_lut[LUT_ENTRIES_PER_QUERY];  // 512 floats = 2048 bytes

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) return;

    // Cooperative load of this query's LUT into shared memory
    const float* lut_ptr = d_lut + query_idx * LUT_ENTRIES_PER_QUERY;
    #pragma unroll
    for (int i = threadIdx.x; i < LUT_ENTRIES_PER_QUERY; i += blockDim.x) {
        smem_lut[i] = lut_ptr[i];
    }
    __syncthreads();

    const float cb1_sumq = d_cb1_sumq[query_idx];
    const size_t pair_start = d_pair_offsets[query_idx];
    const size_t pair_end   = d_pair_offsets[query_idx + 1];
    const size_t num_embs   = pair_end - pair_start;

    for (size_t idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         idx < num_embs;
         idx += (size_t)blockDim.x * gridDim.x)
    {
        const size_t emb_id = d_emb_ids[pair_start + idx];
        const uint64_t* code_ptr =
            (const uint64_t*)(d_one_bit_code + emb_id * CODE_BYTES);
        uint64_t code_regs[NUM_U64];
        code_regs[0] = code_ptr[0];
        code_regs[1] = code_ptr[1];

        float ip = 0.0f;
        #pragma unroll
        for (int blk = 0; blk < NUM_U64; blk++) {
            uint64_t code = code_regs[blk];
            #pragma unroll
            for (int n = 0; n < 16; n++) {
                int nibble = (code >> (n * 4)) & 0xF;
                int chunk_idx = blk * 16 + n;
                ip += smem_lut[chunk_idx * LUT_SIZE + nibble];
            }
        }

        float dist = (ip - cb1_sumq) * d_one_bit_factor[emb_id];
        d_out_dists[query_idx * max_embs_per_query + idx] = dist;
    }
}

__global__ void stage2_binary_ip_lut_kernel(
    const float* __restrict__ d_lut,           // [Q_DOCLEN * LUT_ENTRIES_PER_QUERY]
    const char*  __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_token_ids,
    float* __restrict__ d_out_dists,
    size_t total_tokens,
    size_t batch_tokens
) {
    extern __shared__ float smem[];
    float* smem_lut = smem;                              // [TILE_Q * 512]
    float* smem_cb1_sumq = smem + STAGE2_LUT_SMEM_FLOATS;  // [TILE_Q]

    size_t tok_base = (size_t)blockIdx.x * blockDim.x;

    for (size_t tok_batch_start = tok_base;
         tok_batch_start < batch_tokens;
         tok_batch_start += (size_t)blockDim.x * gridDim.x)
    {
        size_t tok_idx = tok_batch_start + threadIdx.x;
        bool valid = tok_idx < batch_tokens;

        size_t token_id = 0;
        float factor = 0.0f;
        int nibbles[NUM_CHUNKS];

        if (valid) {
            token_id = d_token_ids[tok_idx];
            factor = d_one_bit_factor[token_id];

            const uint64_t* code_ptr =
                (const uint64_t*)(d_one_bit_code + token_id * CODE_BYTES);
            uint64_t code_regs[NUM_U64];
            code_regs[0] = code_ptr[0];
            code_regs[1] = code_ptr[1];

            #pragma unroll
            for (int blk = 0; blk < NUM_U64; blk++) {
                uint64_t code = code_regs[blk];
                #pragma unroll
                for (int n = 0; n < 16; n++) {
                    nibbles[blk * 16 + n] = (code >> (n * 4)) & 0xF;
                }
            }
        }

        #pragma unroll
        for (int tile = 0; tile < STAGE2_LUT_NUM_TILES; tile++) {
            int q_start = tile * STAGE2_LUT_TILE_Q;

            __syncthreads();
            const float* tile_lut_src = d_lut + q_start * LUT_ENTRIES_PER_QUERY;
            for (int i = threadIdx.x; i < STAGE2_LUT_SMEM_FLOATS; i += blockDim.x) {
                smem_lut[i] = tile_lut_src[i];
            }
            if (threadIdx.x < STAGE2_LUT_TILE_Q) {
                smem_cb1_sumq[threadIdx.x] = d_cb1_sumq[q_start + threadIdx.x];
            }
            __syncthreads();

            if (valid) {
                #pragma unroll
                for (int tq = 0; tq < STAGE2_LUT_TILE_Q; tq++) {
                    const float* q_lut = smem_lut + tq * LUT_ENTRIES_PER_QUERY;
                    float ip = 0.0f;

                    #pragma unroll
                    for (int c = 0; c < NUM_CHUNKS; c++) {
                        ip += q_lut[c * LUT_SIZE + nibbles[c]];
                    }

                    int q = q_start + tq;
                    d_out_dists[(size_t)q * total_tokens + tok_idx] =
                        (ip - smem_cb1_sumq[tq]) * factor;
                }
            }
        }
    }
}

namespace {

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFF, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }
    return value;
}

}  // namespace

namespace gpu_mvr::v1 {

__global__ void doc_score(
    const float*  __restrict__ d_token_dists,  // [Q_DOCLEN][total_tokens]
    const size_t* __restrict__ d_candidate_offsets,
    float*        d_doc_scores,
    size_t total_tokens,
    size_t num_candidates
) {
    constexpr int TILE_T = 8;
    __shared__ float tile[Q_DOCLEN * TILE_T];
    __shared__ float max_vals[Q_DOCLEN];
    __shared__ float reduce_buf[256];  // Assuming max 256 threads per block

    size_t cand_idx = blockIdx.x;
    if (cand_idx >= num_candidates) return;

    size_t tok_start = d_candidate_offsets[cand_idx];
    size_t tok_end = d_candidate_offsets[cand_idx + 1];
    size_t num_tokens = tok_end - tok_start;

    for (int j = threadIdx.x; j < Q_DOCLEN; j += blockDim.x) {
        max_vals[j] = -FLT_MAX;
    }
    __syncthreads();

    for (size_t t_base = 0; t_base < num_tokens; t_base += TILE_T) {
        int tile_size = ((size_t)TILE_T < num_tokens - t_base) ? TILE_T : (int)(num_tokens - t_base);

        __syncthreads();
        for (int idx = threadIdx.x; idx < Q_DOCLEN * tile_size; idx += blockDim.x) {
            int q = idx / tile_size;
            int t_local = idx % tile_size;
            tile[q * TILE_T + t_local] = d_token_dists[q * total_tokens + tok_start + t_base + t_local];
        }
        __syncthreads();

        for (size_t j = threadIdx.x; j < Q_DOCLEN; j += blockDim.x) {
            float local_max = max_vals[j];
            for (int t_local = 0; t_local < tile_size; t_local++) {
                local_max = fmaxf(local_max, tile[j * TILE_T + t_local]);
            }
            max_vals[j] = local_max;
        }
    }
    __syncthreads();

    float my_sum = 0.0f;
    for (size_t j = threadIdx.x; j < Q_DOCLEN; j += blockDim.x) {
        my_sum += max_vals[j];
    }

    reduce_buf[threadIdx.x] = my_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_doc_scores[cand_idx] = reduce_buf[0];
    }
}

void launch_doc_score(
    const float* d_token_dists,
    const size_t* d_candidate_offsets,
    float* d_doc_scores,
    float*,
    size_t total_tokens,
    size_t num_candidates,
    int score_threads,
    cudaStream_t stream
) {
    if (num_candidates == 0) return;
    doc_score<<<num_candidates, score_threads, 0, stream>>>(
        d_token_dists, d_candidate_offsets, d_doc_scores, total_tokens, num_candidates);
}

}  // namespace gpu_mvr::v1

namespace gpu_mvr::v2 {

__global__ void doc_score_max(
    const float* __restrict__ d_token_dists,
    const size_t* __restrict__ d_candidate_offsets,
    float* __restrict__ d_doc_query_max,
    size_t total_tokens,
    size_t num_candidates
) {
    const size_t cand_idx = blockIdx.x;
    const int q_idx = (int)blockIdx.y;
    if (cand_idx >= num_candidates || q_idx >= Q_DOCLEN) return;

    const size_t tok_start = d_candidate_offsets[cand_idx];
    const size_t tok_end = d_candidate_offsets[cand_idx + 1];
    float local_max = -FLT_MAX;

    for (size_t tok = tok_start + threadIdx.x; tok < tok_end; tok += blockDim.x) {
        local_max = fmaxf(local_max, d_token_dists[(size_t)q_idx * total_tokens + tok]);
    }

    local_max = warp_reduce_max(local_max);

    __shared__ float warp_max[8];
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    float block_max = -FLT_MAX;
    if (warp_id == 0) {
        const int warp_count = blockDim.x / warpSize;
        block_max = (lane < warp_count) ? warp_max[lane] : -FLT_MAX;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) {
            d_doc_query_max[cand_idx * Q_DOCLEN + q_idx] = block_max;
        }
    }
}

__global__ void doc_score(
    const float* __restrict__ d_doc_query_max,
    float* __restrict__ d_doc_scores,
    size_t num_candidates
) {
    const size_t cand_idx = blockIdx.x;
    if (cand_idx >= num_candidates) return;

    float partial = 0.0f;
    for (int q = threadIdx.x; q < Q_DOCLEN; q += blockDim.x) {
        partial += d_doc_query_max[cand_idx * Q_DOCLEN + q];
    }

    partial = warp_reduce_sum(partial);

    __shared__ float warp_sums[1];
    if (threadIdx.x == 0) {
        warp_sums[0] = partial;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        d_doc_scores[cand_idx] = warp_sums[0];
    }
}

void launch_doc_score(
    const float* d_token_dists,
    const size_t* d_candidate_offsets,
    float* d_doc_scores,
    float* d_doc_query_max,
    size_t total_tokens,
    size_t num_candidates,
    int score_threads,
    cudaStream_t stream
) {
    if (num_candidates == 0) return;

    // The legacy launch shape defaults to 128 threads, but the v2 max kernel
    // often processes relatively short per-document token spans. Using 64
    // threads reduces overprovisioned threads while keeping enough warps to
    // hide memory latency on Ada/Ampere-class GPUs.
    int reduce_threads = score_threads > 0 ? score_threads : 128;
    reduce_threads = reduce_threads > 64 ? 64 : reduce_threads;
    reduce_threads = reduce_threads < 32 ? 32 : reduce_threads;
    dim3 grid((unsigned int)num_candidates, Q_DOCLEN);
    doc_score_max<<<grid, reduce_threads, 0, stream>>>(
        d_token_dists, d_candidate_offsets, d_doc_query_max, total_tokens, num_candidates);
    doc_score<<<num_candidates, 32, 0, stream>>>(
        d_doc_query_max, d_doc_scores, num_candidates);
}

}  // namespace gpu_mvr::v2

namespace gpu_mvr {

void configure_binary_ip_kernels() {
#if GPU_MVR_BINARY_IP_IMPL == 2
    size_t stage2_lut_smem = STAGE2_LUT_SMEM_FLOATS * sizeof(float) + STAGE2_LUT_TILE_Q * sizeof(float);
    cudaError_t err = cudaFuncSetAttribute(
        stage2_binary_ip_lut_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        stage2_lut_smem);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
#endif
}

void precompute_binary_ip_state(
    const float* d_queries,
    float* d_lut,
    cudaStream_t stream
) {
#if GPU_MVR_BINARY_IP_IMPL == 2
    precompute_lut_kernel<<<Q_DOCLEN, 256, 0, stream>>>(d_queries, d_lut);
#else
    (void)d_queries;
    (void)d_lut;
    (void)stream;
#endif
}

void launch_stage1_binary_ip(
    const float* d_queries,
    const float* d_lut,
    const char* d_one_bit_code,
    const float* d_one_bit_factor,
    const float* d_cb1_sumq,
    const size_t* d_emb_ids,
    const int* d_pair_offsets,
    float* d_out_dists,
    size_t max_embs_per_query,
    int threads_per_block,
    cudaStream_t stream
) {
    dim3 grid((unsigned int)((max_embs_per_query + threads_per_block - 1) / threads_per_block), Q_DOCLEN);
#if GPU_MVR_BINARY_IP_IMPL == 2
    stage1_binary_ip_lut_kernel<<<grid, threads_per_block, 0, stream>>>(
        d_lut, d_one_bit_code, d_one_bit_factor, d_cb1_sumq,
        d_emb_ids, d_pair_offsets, d_out_dists, max_embs_per_query);
#else
    stage1_binary_ip_kernel_v2<<<grid, threads_per_block, 0, stream>>>(
        d_queries, d_one_bit_code, d_one_bit_factor, d_cb1_sumq,
        d_emb_ids, d_pair_offsets, d_out_dists, max_embs_per_query);
#endif
}

void launch_stage2_binary_ip(
    const float* d_queries,
    const float* d_lut,
    const char* d_one_bit_code,
    const float* d_one_bit_factor,
    const float* d_cb1_sumq,
    const size_t* d_token_ids,
    float* d_out_dists,
    size_t total_tokens,
    size_t batch_tokens,
    int threads_per_block,
    cudaStream_t stream
) {
    int blocks_x = (int)((batch_tokens + threads_per_block - 1) / threads_per_block);
#if GPU_MVR_BINARY_IP_IMPL == 2
    size_t stage2_lut_smem = STAGE2_LUT_SMEM_FLOATS * sizeof(float) + STAGE2_LUT_TILE_Q * sizeof(float);
    stage2_binary_ip_lut_kernel<<<blocks_x, threads_per_block, stage2_lut_smem, stream>>>(
        d_lut, d_one_bit_code, d_one_bit_factor, d_cb1_sumq,
        d_token_ids, d_out_dists, total_tokens, batch_tokens);
#else
    stage2_binary_ip_kernel_v2<<<blocks_x, threads_per_block, 0, stream>>>(
        d_queries, d_one_bit_code, d_one_bit_factor, d_cb1_sumq,
        d_token_ids, d_out_dists, total_tokens, batch_tokens);
#endif
}

void launch_doc_score(
    const float* d_token_dists,
    const size_t* d_candidate_offsets,
    float* d_doc_scores,
    float* d_doc_query_max,
    size_t total_tokens,
    size_t num_candidates,
    int score_threads,
    cudaStream_t stream
) {
#if GPU_MVR_DOC_SCORE_IMPL == 1
    v1::launch_doc_score(
        d_token_dists, d_candidate_offsets, d_doc_scores, d_doc_query_max,
        total_tokens, num_candidates, score_threads, stream);
#else
    v2::launch_doc_score(
        d_token_dists, d_candidate_offsets, d_doc_scores, d_doc_query_max,
        total_tokens, num_candidates, score_threads, stream);
#endif
}

}  // namespace gpu_mvr

__global__ void extract_one_bit_dists_kernel(
    const float*  __restrict__ d_token_dists,     // [Q_DOCLEN][total_tokens] GPU layout
    const size_t* __restrict__ d_candidate_offsets,
    const int*    __restrict__ d_selected_indices,
    float*        d_out_one_bit_dists,             // [token][query] output layout
    const size_t* __restrict__ d_out_offsets,
    size_t total_tokens,
    size_t k
) {
    size_t sel_idx = blockIdx.x;
    if (sel_idx >= k) return;

    size_t cand_idx = d_selected_indices[sel_idx];
    size_t tok_start = d_candidate_offsets[cand_idx];
    size_t tok_end = d_candidate_offsets[cand_idx + 1];
    size_t num_tokens = tok_end - tok_start;
    size_t out_base = d_out_offsets[sel_idx];
    size_t total_elems = num_tokens * Q_DOCLEN;

    // Iterate over output elements linearly for coalesced writes
    // Output layout: [token][query] → out[(out_base + t) * Q_DOCLEN + q]
    // Adjacent threads write adjacent addresses when iterating linearly
    for (size_t i = threadIdx.x; i < total_elems; i += blockDim.x) {
        size_t t_local = i / Q_DOCLEN;
        size_t q_idx = i - t_local * Q_DOCLEN;  // avoid expensive modulo

        // Read from [query][token] layout
        float val = d_token_dists[q_idx * total_tokens + tok_start + t_local];
        // Write to [token][query] layout
        d_out_one_bit_dists[(out_base + t_local) * Q_DOCLEN + q_idx] = val;
    }
}

__global__ void gather_token_ids_kernel(
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

/**
 * Compute per-query expansion sizes for cluster ID expansion.
 * Each thread handles one query and sums up cluster sizes.
 */
__global__ void compute_query_expansion_sizes_kernel(
    const uint32_t* __restrict__ d_cagra_labels,  // [Q_DOCLEN * nprobe]
    const size_t*       __restrict__ d_cluster_pos,   // [n_clusters + 1]
    int*                d_query_sizes,                // [Q_DOCLEN] output
    int                 nprobe,
    size_t              n_clusters,
    size_t              num_queries                   // Q_DOCLEN
) {
    size_t query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;

    int total_size = 0;
    for (int p = 0; p < nprobe; ++p) {
        uint32_t cluster_id = d_cagra_labels[query_idx * nprobe + p];
        if (cluster_id < 0 || cluster_id >= n_clusters) continue;

        size_t cluster_size = d_cluster_pos[cluster_id + 1] - d_cluster_pos[cluster_id];
        total_size += cluster_size;
    }

    d_query_sizes[query_idx] = total_size;
}

/**
 * Expand cluster IDs to embedding IDs on GPU.
 * One block per query, threads cooperatively expand all clusters for that query.
 */
__global__ void expand_cluster_ids_kernel(
    const uint32_t* __restrict__ d_cagra_labels,  // [Q_DOCLEN * nprobe]
    const int*          __restrict__ d_inv_list,      // [inv_list_size]
    const size_t*       __restrict__ d_cluster_pos,   // [n_clusters + 1]
    const int*          __restrict__ d_query_offsets, // [Q_DOCLEN + 1] prefix sum
    size_t*             d_emb_ids,                    // [total_pairs] output
    int                 nprobe,
    size_t              n_clusters,
    size_t              num_queries                   // Q_DOCLEN
) {
    size_t query_idx = blockIdx.x;
    if (query_idx >= num_queries) return;

    size_t out_start = d_query_offsets[query_idx];
    size_t write_pos = 0;

    // Process each probe for this query
    for (int p = 0; p < nprobe; ++p) {
        uint32_t cluster_id = d_cagra_labels[query_idx * nprobe + p];
        if (cluster_id < 0 || cluster_id >= n_clusters) continue;

        size_t cluster_start = d_cluster_pos[cluster_id];
        size_t cluster_end = d_cluster_pos[cluster_id + 1];
        size_t cluster_size = cluster_end - cluster_start;

        // Cooperative copy: all threads in block write cluster embeddings
        for (size_t i = threadIdx.x; i < cluster_size; i += blockDim.x) {
            d_emb_ids[out_start + write_pos + i] = d_inv_list[cluster_start + i];
        }

        write_pos += cluster_size;
        __syncthreads();  // Ensure all threads finish before next cluster
    }
}

/**
 * Gather document pointers for top-k docs to compute token counts.
 * Each thread handles one document.
 */
__global__ void gather_doc_lengths_kernel(
    const int* __restrict__ d_topk_doc_ids,
    const int* __restrict__ d_doc_ptrs,
    int*       d_doc_lengths,
    size_t k
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    
    int doc_id = d_topk_doc_ids[idx];
    int doc_len = d_doc_ptrs[doc_id + 1] - d_doc_ptrs[doc_id];
    d_doc_lengths[idx] = doc_len;
}

// Atomic max for floats using compare-and-swap
__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                       __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/**
 * Atomic max-pooling and aggregation for Stage 1 with touched-doc tracking.
 * Each (emb_id, query_idx) → (doc_id, query_idx) → atomic max into doc_query_matrix.
 * Simultaneously builds a compact list of unique touched document IDs for sparse
 * downstream processing (sum, sort, cleanup).
 */
__global__ void aggregate_stage1_tracked_kernel(
    const size_t* __restrict__ d_emb_ids,
    const float*  __restrict__ d_emb_dists,       // [Q_DOCLEN * max_embs_per_query] layout
    const int*    __restrict__ d_pair_offsets,
    const int*    __restrict__ d_doc_ids,
    float*        d_doc_query_max,                 // [Q_DOCLEN * num_docs] transposed matrix
    int*          d_doc_touched,                   // [num_docs] per-doc flag
    int*          d_touched_doc_list,              // [num_docs] compact output list
    int*          d_num_touched_docs,              // [1] atomic counter
    size_t        num_docs,
    size_t        total_pairs,
    size_t        max_embs_per_query
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    // Binary search for query index (log2(Q_DOCLEN) iterations instead of linear scan)
    int q_idx;
    {
        int lo = 0, hi = Q_DOCLEN;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if ((size_t)d_pair_offsets[mid + 1] <= idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        q_idx = lo;
    }

    size_t emb_id = d_emb_ids[idx];
    int doc_id = d_doc_ids[emb_id];

    int local_pair_idx = (int)(idx - (size_t)d_pair_offsets[q_idx]);
    float dist = d_emb_dists[(size_t)q_idx * max_embs_per_query + local_pair_idx];

    if (doc_id >= (int)num_docs) return;

    // Atomic max into transposed matrix [q_idx * num_docs + doc_id]
    size_t matrix_idx = (size_t)q_idx * num_docs + doc_id;
    atomicMaxFloat(&d_doc_query_max[matrix_idx], dist);

    // Track touched docs: read flag first (cheap L2 read), atomicExch only if unseen
    if (d_doc_touched[doc_id] == 0) {
        int was_touched = atomicExch(&d_doc_touched[doc_id], 1);
        if (was_touched == 0) {
            int pos = atomicAdd(d_num_touched_docs, 1);
            d_touched_doc_list[pos] = doc_id;
        }
    }
}

/**
 * Sum max distances across queries for each document.
 */
__global__ void sum_doc_scores_kernel(
    const float* __restrict__ d_doc_query_max,  // [num_docs * Q_DOCLEN]
    float*       d_doc_scores,                   // [num_docs]
    size_t       num_docs
) {
    size_t doc_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (doc_id >= num_docs) return;

    float score = 0.0f;
    #pragma unroll
    for (size_t q = 0; q < Q_DOCLEN; ++q) {
        score += d_doc_query_max[q * num_docs + doc_id];  // transposed: coalesced reads
    }
    d_doc_scores[doc_id] = score;
}

/**
 * Sum max distances across queries for only the touched (active) documents.
 * Writes compact score and doc_id arrays suitable for direct CUB sort.
 */
__global__ void sum_doc_scores_sparse_kernel(
    const float* __restrict__ d_doc_query_max,    // [Q_DOCLEN * num_docs] transposed
    const int*   __restrict__ d_touched_doc_list, // [num_touched] compact doc IDs
    float*       d_scores_out,                     // [num_touched] compact scores
    int*         d_doc_ids_out,                    // [num_touched] compact doc IDs for sort
    int          num_touched,
    size_t       num_docs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_touched) return;

    int doc_id = d_touched_doc_list[idx];
    float score = 0.0f;
    #pragma unroll
    for (int q = 0; q < Q_DOCLEN; ++q) {
        score += d_doc_query_max[(size_t)q * num_docs + doc_id];
    }
    d_scores_out[idx] = score;
    d_doc_ids_out[idx] = doc_id;
}

/**
 * Reset doc_query_max entries and touched flags for only the documents that were
 * active in the current search.  This replaces the expensive full-matrix memset.
 */
__global__ void cleanup_touched_docs_kernel(
    float*       d_doc_query_max,                  // [Q_DOCLEN * num_docs] transposed
    int*         d_doc_touched,                    // [num_docs] per-doc flag
    const int*   __restrict__ d_touched_doc_list,  // [num_touched] compact doc IDs
    int          num_touched,
    size_t       num_docs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_touched) return;

    int doc_id = d_touched_doc_list[idx];
    #pragma unroll
    for (int q = 0; q < Q_DOCLEN; ++q) {
        d_doc_query_max[(size_t)q * num_docs + doc_id] = 0.0f;
    }
    d_doc_touched[doc_id] = 0;
}
