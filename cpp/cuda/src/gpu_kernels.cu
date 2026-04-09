#include "gpu_kernels.cuh"

#include <cfloat>

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
    __shared__ float smem_query[PADDED_DIM];

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) return;

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
            uint64_t bits = code_regs[blk];
            int base = blk * 64;
            while (bits) {
                int pos = __ffsll(bits) - 1;
                ip += smem_query[base + pos];
                bits &= bits - 1;
            }
        }
        float dist = (ip - cb1_sumq) * d_one_bit_factor[emb_id];

        d_out_dists[query_idx * max_embs_per_query + idx] = dist;
    }
}

__global__ void stage2_binary_ip_kernel_v2(
    const float* __restrict__ d_queries,
    const char*  __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_token_ids,
    float* __restrict__ d_out_dists,
    size_t total_tokens,
    size_t batch_tokens
) {
    __shared__ float smem_queries[Q_DOCLEN * PADDED_DIM];
    __shared__ float smem_cb1_sumq[Q_DOCLEN];

    const int total_query_floats = Q_DOCLEN * PADDED_DIM;
    for (int i = threadIdx.x; i < total_query_floats; i += blockDim.x) {
        smem_queries[i] = d_queries[i];
    }
    if (threadIdx.x < Q_DOCLEN) {
        smem_cb1_sumq[threadIdx.x] = d_cb1_sumq[threadIdx.x];
    }
    __syncthreads();

    for (size_t tok_idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         tok_idx < batch_tokens;
         tok_idx += (size_t)blockDim.x * gridDim.x)
    {
        const size_t token_id = d_token_ids[tok_idx];
        const float factor = d_one_bit_factor[token_id];

        const uint64_t* code_ptr =
            (const uint64_t*)(d_one_bit_code + token_id * CODE_BYTES);
        uint64_t code_regs[NUM_U64];
        code_regs[0] = code_ptr[0];
        code_regs[1] = code_ptr[1];

        #pragma unroll
        for (int q = 0; q < Q_DOCLEN; q++) {
            const float* q_smem = smem_queries + q * PADDED_DIM;
            const float cb1_sumq = smem_cb1_sumq[q];

            float ip = 0.0f;
            #pragma unroll
            for (int blk = 0; blk < NUM_U64; blk++) {
                uint64_t bits = code_regs[blk];
                int base = blk * 64;
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

__global__ void precompute_lut_kernel(
    const float* __restrict__ d_queries,
    float* __restrict__ d_lut
) {
    const int query_idx = blockIdx.x;
    if (query_idx >= Q_DOCLEN) return;

    const float* q_ptr = d_queries + query_idx * PADDED_DIM;
    float* lut_ptr = d_lut + query_idx * LUT_ENTRIES_PER_QUERY;

    for (int idx = threadIdx.x; idx < LUT_ENTRIES_PER_QUERY; idx += blockDim.x) {
        int chunk_idx = idx / LUT_SIZE;
        int lut_entry = idx % LUT_SIZE;

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
    const float* __restrict__ d_lut,
    const char*  __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_emb_ids,
    const int*   __restrict__ d_pair_offsets,
    float* __restrict__ d_out_dists,
    size_t max_embs_per_query
) {
    __shared__ float smem_lut[LUT_ENTRIES_PER_QUERY];

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) return;

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
    const float* __restrict__ d_lut,
    const char*  __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_token_ids,
    float* __restrict__ d_out_dists,
    size_t total_tokens,
    size_t batch_tokens
) {
    extern __shared__ float smem[];
    float* smem_lut = smem;
    float* smem_cb1_sumq = smem + STAGE2_LUT_SMEM_FLOATS;

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
                    d_out_dists[q * total_tokens + tok_idx] =
                        (ip - smem_cb1_sumq[tq]) * factor;
                }
            }
        }
    }
}

__global__ void doc_score_kernel(
    const float*  __restrict__ d_token_dists,
    const size_t* __restrict__ d_candidate_offsets,
    float*        d_doc_scores,
    size_t total_tokens,
    size_t num_candidates
) {
    constexpr int TILE_T = 8;
    __shared__ float tile[Q_DOCLEN * TILE_T];
    __shared__ float max_vals[Q_DOCLEN];
    __shared__ float reduce_buf[256];

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

__global__ void extract_one_bit_dists_kernel(
    const float*  __restrict__ d_token_dists,
    const size_t* __restrict__ d_candidate_offsets,
    const int*    __restrict__ d_selected_indices,
    float*        d_out_one_bit_dists,
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

    for (size_t i = threadIdx.x; i < total_elems; i += blockDim.x) {
        size_t t_local = i / Q_DOCLEN;
        size_t q_idx = i - t_local * Q_DOCLEN;

        float val = d_token_dists[q_idx * total_tokens + tok_start + t_local];
        d_out_one_bit_dists[(out_base + t_local) * Q_DOCLEN + q_idx] = val;
    }
}

__global__ void extract_one_bit_dists_kernel_v2(
    const float*  __restrict__ d_token_dists,
    const size_t* __restrict__ d_candidate_offsets,
    const int*    __restrict__ d_selected_indices,
    float*        h_pinned_out,
    const size_t* __restrict__ d_out_offsets,
    size_t total_tokens,
    size_t k
) {
    constexpr size_t VEC_SIZE = 4;
    constexpr size_t Q_DOCLEN_VEC = Q_DOCLEN / VEC_SIZE;

    for (size_t sel_idx = 0; sel_idx < k; ++sel_idx) {
        size_t cand_idx = d_selected_indices[sel_idx];
        size_t tok_start = d_candidate_offsets[cand_idx];
        size_t tok_end = d_candidate_offsets[cand_idx + 1];
        size_t num_tokens = tok_end - tok_start;
        size_t out_base = d_out_offsets[sel_idx];
        size_t total_vec_elems = num_tokens * Q_DOCLEN_VEC;

        for (size_t vi = threadIdx.x; vi < total_vec_elems; vi += blockDim.x) {
            size_t t_local = vi / Q_DOCLEN_VEC;
            size_t vec_idx = vi - t_local * Q_DOCLEN_VEC;
            size_t q_base = vec_idx * VEC_SIZE;

            float4 packed;
            packed.x = d_token_dists[(q_base + 0) * total_tokens + tok_start + t_local];
            packed.y = d_token_dists[(q_base + 1) * total_tokens + tok_start + t_local];
            packed.z = d_token_dists[(q_base + 2) * total_tokens + tok_start + t_local];
            packed.w = d_token_dists[(q_base + 3) * total_tokens + tok_start + t_local];

            float4* out_ptr = (float4*)(h_pinned_out + (out_base + t_local) * Q_DOCLEN + q_base);
            *out_ptr = packed;
        }
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

__global__ void compute_query_expansion_sizes_kernel(
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
        total_size += cluster_size;
    }

    d_query_sizes[query_idx] = total_size;
}

__global__ void expand_cluster_ids_kernel(
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

__global__ void aggregate_stage1_tracked_kernel(
    const size_t* __restrict__ d_emb_ids,
    const float*  __restrict__ d_emb_dists,
    const int*    __restrict__ d_pair_offsets,
    const int*    __restrict__ d_doc_ids,
    float*        d_doc_query_max,
    int*          d_doc_touched,
    int*          d_touched_doc_list,
    int*          d_num_touched_docs,
    size_t        num_docs,
    size_t        total_pairs,
    size_t        max_embs_per_query
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

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

    size_t matrix_idx = (size_t)q_idx * num_docs + doc_id;
    atomicMaxFloat(&d_doc_query_max[matrix_idx], dist);

    if (d_doc_touched[doc_id] == 0) {
        int was_touched = atomicExch(&d_doc_touched[doc_id], 1);
        if (was_touched == 0) {
            int pos = atomicAdd(d_num_touched_docs, 1);
            d_touched_doc_list[pos] = doc_id;
        }
    }
}

__global__ void sum_doc_scores_kernel(
    const float* __restrict__ d_doc_query_max,
    float*       d_doc_scores,
    size_t       num_docs
) {
    size_t doc_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (doc_id >= num_docs) return;

    float score = 0.0f;
    #pragma unroll
    for (size_t q = 0; q < Q_DOCLEN; ++q) {
        score += d_doc_query_max[q * num_docs + doc_id];
    }
    d_doc_scores[doc_id] = score;
}

__global__ void sum_doc_scores_sparse_kernel(
    const float* __restrict__ d_doc_query_max,
    const int*   __restrict__ d_touched_doc_list,
    float*       d_scores_out,
    int*         d_doc_ids_out,
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
