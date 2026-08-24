#include "chimera/kernels.cuh"

#include <cfloat>
#include <cub/cub.cuh>

namespace Chimera {

namespace {

__device__ __forceinline__ float warp_group_max(
    float value,
    unsigned int group_mask,
    unsigned int active_mask
) {
    const int lane = threadIdx.x & 31;
    unsigned int remaining = group_mask & ~(1u << lane);
    while (remaining != 0u) {
        const int peer_lane = __ffs(remaining) - 1;
        value = fmaxf(value, __shfl_sync(active_mask, value, peer_lane));
        remaining &= (remaining - 1);
    }
    return value;
}

__device__ __forceinline__ unsigned int warp_matching_doc_mask(
    int doc_id,
    unsigned int active_mask
) {
    unsigned int match_mask = 0u;
    #pragma unroll
    for (int lane = 0; lane < 32; ++lane) {
        const unsigned int lane_bit = 1u << lane;
        if ((active_mask & lane_bit) == 0u) {
            continue;
        }
        if (__shfl_sync(active_mask, doc_id, lane) == doc_id) {
            match_mask |= lane_bit;
        }
    }
    return match_mask;
}

__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_int,
            assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ void mark_candidate_document(
    candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    int doc_id
) {
    const int bucket = doc_id / kCandidateBitmapBucketWidth;
    const int bit = doc_id % kCandidateBitmapBucketWidth;
    atomicOr(&d_candidate_bitmap[bucket], candidate_bitmap_bucket_t(1u) << bit);
}

__device__ __forceinline__ int candidate_dense_index(
    const candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    const candidate_bitmap_offset_t* __restrict__ d_candidate_bitmap_offsets,
    int doc_id
) {
    const int bucket = doc_id / kCandidateBitmapBucketWidth;
    const int group = bucket / kCandidateBitmapCompressionRatio;
    const int group_bucket_start = group * kCandidateBitmapCompressionRatio;

    int dense_index = static_cast<int>(d_candidate_bitmap_offsets[group]);
    #pragma unroll
    for (int b = 0; b < kCandidateBitmapCompressionRatio; ++b) {
        const int bucket_idx = group_bucket_start + b;
        if (bucket_idx >= bucket) {
            break;
        }
        dense_index += __popc(__ldg(&d_candidate_bitmap[bucket_idx]));
    }

    const int bit = doc_id % kCandidateBitmapBucketWidth;
    const candidate_bitmap_bucket_t bucket_bits = __ldg(&d_candidate_bitmap[bucket]);
    const candidate_bitmap_bucket_t prior_bits =
        (bit == 0)
            ? candidate_bitmap_bucket_t(0)
            : (bucket_bits & ((candidate_bitmap_bucket_t(1u) << bit) - 1u));
    dense_index += __popc(prior_bits);
    return dense_index;
}

__device__ __forceinline__ uint32_t cluster_virtual_to_token_position(
    uint32_t cluster_start,
    int cluster_size,
    int virtual_local_idx
) {
    const int misalign = static_cast<int>(cluster_start & (kCandidateRefinementClusterAlignment - 1));
    const int head = (misalign == 0)
        ? 0
        : static_cast<int>(kCandidateRefinementClusterAlignment) - misalign;
    if (head == 0 || head >= cluster_size) {
        return cluster_start + static_cast<uint32_t>(virtual_local_idx);
    }

    const int aligned_run = cluster_size - head;
    if (virtual_local_idx < aligned_run) {
        return cluster_start + static_cast<uint32_t>(head + virtual_local_idx);
    }
    return cluster_start + static_cast<uint32_t>(virtual_local_idx - aligned_run);
}

}  // namespace

__global__ void count_candidate_bitmap_kernel(
    const candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    size_t num_buckets,
    candidate_bitmap_offset_t* __restrict__ d_candidate_bitmap_offsets
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_buckets) {
        return;
    }

    if ((idx % kCandidateBitmapCompressionRatio) == 0) {
        candidate_bitmap_offset_t count = 0;
        const size_t group_bucket_start = idx;
        #pragma unroll
        for (int offset = 0; offset < kCandidateBitmapCompressionRatio; ++offset) {
            const size_t bucket_idx = group_bucket_start + offset;
            if (bucket_idx >= num_buckets) {
                break;
            }
            count += __popc(d_candidate_bitmap[bucket_idx]);
        }
        d_candidate_bitmap_offsets[idx / kCandidateBitmapCompressionRatio] = count;
    }
}

__global__ void materialize_candidate_documents_kernel(
    const candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    size_t num_buckets,
    const candidate_bitmap_offset_t* __restrict__ d_candidate_bitmap_offsets,
    int* __restrict__ d_candidate_doc_ids
) {
    const size_t bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= num_buckets) {
        return;
    }

    const size_t group_idx = bucket_idx / kCandidateBitmapCompressionRatio;
    const size_t group_bucket_start = group_idx * kCandidateBitmapCompressionRatio;
    uint32_t out_idx = d_candidate_bitmap_offsets[group_idx];

    #pragma unroll
    for (int offset = 0; offset < kCandidateBitmapCompressionRatio; ++offset) {
        const size_t prev_bucket = group_bucket_start + offset;
        if (prev_bucket >= bucket_idx) {
            break;
        }
        out_idx += __popc(__ldg(&d_candidate_bitmap[prev_bucket]));
    }

    candidate_bitmap_bucket_t bits = __ldg(&d_candidate_bitmap[bucket_idx]);
    while (bits != 0u) {
        const int bit = __ffs(bits) - 1;
        d_candidate_doc_ids[out_idx++] =
            static_cast<int>(bucket_idx * kCandidateBitmapBucketWidth + bit);
        bits &= (bits - 1);
    }
}

__global__ void sum_partial_scores_kernel(
    const float* __restrict__ d_partial_query_scores,
    const int*   __restrict__ d_candidate_doc_ids,
    float*       d_partial_scores,
    int*         d_scored_doc_ids,
    int          num_candidate_docs
) {
    const int warp_id_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int global_warp_id =
        blockIdx.x * PARTIAL_SCORE_WARPS_PER_BLOCK + warp_id_in_block;

    if (global_warp_id >= num_candidate_docs) {
        return;
    }

    const int doc_id = d_candidate_doc_ids[global_warp_id];
    float val =
        d_partial_query_scores[static_cast<size_t>(global_warp_id) * Q_DOCLEN + lane];

    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (lane == 0) {
        d_partial_scores[global_warp_id] = val;
        d_scored_doc_ids[global_warp_id] = doc_id;
    }
}

__global__ void mark_candidate_documents_kernel(
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    size_t          num_docs,
    int             nprobe,
    size_t          n_clusters
) {
    __shared__ uint32_t smem_cstart[kCandidateRefinementMaxNprobe];
    __shared__ int smem_prefix[kCandidateRefinementMaxNprobe + 1];

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) {
        return;
    }

    const uint32_t* my_labels = d_cagra_labels + query_idx * nprobe;
    for (int p = threadIdx.x; p < nprobe; p += blockDim.x) {
        const uint32_t label = my_labels[p];
        if (label < static_cast<uint32_t>(n_clusters)) {
            const size_t start = d_cluster_pos[label];
            smem_cstart[p] = static_cast<uint32_t>(start);
            smem_prefix[p + 1] = static_cast<int>(d_cluster_pos[label + 1] - start);
        } else {
            smem_cstart[p] = 0;
            smem_prefix[p + 1] = 0;
        }
    }
    if (threadIdx.x == 0) {
        smem_prefix[0] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i <= nprobe; ++i) {
            smem_prefix[i] += smem_prefix[i - 1];
        }
    }
    __syncthreads();

    const int total_elements = smem_prefix[nprobe];

    for (int flat_idx = threadIdx.x + blockIdx.x * blockDim.x;
         flat_idx < total_elements;
         flat_idx += blockDim.x * gridDim.x) {
        int lo = 0;
        int hi = nprobe;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (smem_prefix[mid + 1] <= flat_idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        const int cluster_size = smem_prefix[lo + 1] - smem_prefix[lo];
        const int cluster_local = flat_idx - smem_prefix[lo];
        const uint32_t emb_pos = cluster_virtual_to_token_position(
            smem_cstart[lo], cluster_size, cluster_local);
        const int doc_id = d_clustered_doc_ids[emb_pos];

        if (static_cast<size_t>(doc_id) < num_docs) {
            const unsigned int active_mask = __activemask();
            const unsigned int doc_group = warp_matching_doc_mask(doc_id, active_mask);
            const int lane = threadIdx.x & 31;
            if (lane == (__ffs(doc_group) - 1)) {
                mark_candidate_document(d_candidate_bitmap, doc_id);
            }
        }
    }
}

__global__ void partial_score_kernel(
    const float*    __restrict__ d_lut,
    const char*     __restrict__ d_clustered_code,
    const float*    __restrict__ d_clustered_factor,
    const float*    __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    float*          __restrict__ d_partial_query_scores,
    const candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    const candidate_bitmap_offset_t* __restrict__ d_candidate_bitmap_offsets,
    int             num_candidate_docs,
    size_t          num_docs,
    int             nprobe,
    size_t          n_clusters
) {
    __shared__ float smem_lut[LUT_ENTRIES_PER_QUERY];
    __shared__ uint32_t smem_cstart[kCandidateRefinementMaxNprobe];
    __shared__ int smem_prefix[kCandidateRefinementMaxNprobe + 1];

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) {
        return;
    }

    const float* lut_ptr = d_lut + query_idx * LUT_ENTRIES_PER_QUERY;
    for (int i = threadIdx.x; i < LUT_ENTRIES_PER_QUERY; i += blockDim.x) {
        smem_lut[i] = lut_ptr[i];
    }

    const uint32_t* my_labels = d_cagra_labels + query_idx * nprobe;
    for (int p = threadIdx.x; p < nprobe; p += blockDim.x) {
        const uint32_t label = my_labels[p];
        if (label < static_cast<uint32_t>(n_clusters)) {
            const size_t start = d_cluster_pos[label];
            smem_cstart[p] = static_cast<uint32_t>(start);
            smem_prefix[p + 1] = static_cast<int>(d_cluster_pos[label + 1] - start);
        } else {
            smem_cstart[p] = 0;
            smem_prefix[p + 1] = 0;
        }
    }
    if (threadIdx.x == 0) {
        smem_prefix[0] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i <= nprobe; ++i) {
            smem_prefix[i] += smem_prefix[i - 1];
        }
    }
    __syncthreads();

    const int total_elements = smem_prefix[nprobe];
    const float cb1_sumq = d_cb1_sumq[query_idx];

    for (int flat_idx = threadIdx.x + blockIdx.x * blockDim.x;
         flat_idx < total_elements;
         flat_idx += blockDim.x * gridDim.x) {
        int lo = 0;
        int hi = nprobe;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (smem_prefix[mid + 1] <= flat_idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        const int cluster_size = smem_prefix[lo + 1] - smem_prefix[lo];
        const int cluster_local = flat_idx - smem_prefix[lo];
        const uint32_t emb_pos = cluster_virtual_to_token_position(
            smem_cstart[lo], cluster_size, cluster_local);

        const uint4 code128 = *reinterpret_cast<const uint4*>(
            d_clustered_code + static_cast<size_t>(emb_pos) * CODE_BYTES);
        const float factor = d_clustered_factor[emb_pos];
        const int doc_id = d_clustered_doc_ids[emb_pos];

        float ip = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            ip += smem_lut[n * LUT_SIZE + ((code128.x >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            ip += smem_lut[(8 + n) * LUT_SIZE + ((code128.y >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            ip += smem_lut[(16 + n) * LUT_SIZE + ((code128.z >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            ip += smem_lut[(24 + n) * LUT_SIZE + ((code128.w >> (n * 4)) & 0xF)];

        const float dist = (ip - cb1_sumq) * factor;
        if (dist > 0.0f && static_cast<size_t>(doc_id) < num_docs) {
            const unsigned int active_mask = __activemask();
            const unsigned int doc_group = warp_matching_doc_mask(doc_id, active_mask);
            const int lane = threadIdx.x & 31;
            const int leader_lane = __ffs(doc_group) - 1;
            const float group_max = warp_group_max(dist, doc_group, active_mask);

            if (lane == leader_lane) {
                const int candidate_index = candidate_dense_index(
                    d_candidate_bitmap,
                    d_candidate_bitmap_offsets,
                    doc_id);
                if (candidate_index < 0 ||
                    candidate_index >= num_candidate_docs) {
                    continue;
                }
                atomicMaxFloat(
                    &d_partial_query_scores[
                        static_cast<size_t>(candidate_index) * Q_DOCLEN +
                        query_idx],
                    group_max);
            }
        }
    }
}

__global__ void precompute_one_bit_lut_kernel(
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

__global__ void one_bit_token_score_kernel(
    const float* __restrict__ d_lut,
    const char*  __restrict__ d_clustered_code,
    const float* __restrict__ d_clustered_factor,
    const float* __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_clustered_pos,
    float* __restrict__ d_one_bit_scores,
    size_t total_tokens,
    size_t batch_tokens
) {
    extern __shared__ float smem[];
    float* smem_lut = smem;
    float* smem_cb1_sumq = smem + DOCUMENT_SCORING_LUT_SMEM_FLOATS;

    size_t tok_base = (size_t)blockIdx.x * blockDim.x;

    for (size_t tok_batch_start = tok_base;
         tok_batch_start < batch_tokens;
         tok_batch_start += (size_t)blockDim.x * gridDim.x)
    {
        size_t tok_idx = tok_batch_start + threadIdx.x;
        bool valid = tok_idx < batch_tokens;

        uint32_t clustered_pos = 0;
        float factor = 0.0f;
        int nibbles[NUM_CHUNKS];

        if (valid) {
            clustered_pos = d_clustered_pos[tok_idx];
            factor = d_clustered_factor[clustered_pos];

            const uint64_t* code_ptr =
                (const uint64_t*)(d_clustered_code + static_cast<size_t>(clustered_pos) * CODE_BYTES);
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
        for (int tile = 0; tile < DOCUMENT_SCORING_LUT_NUM_TILES; tile++) {
            int q_start = tile * DOCUMENT_SCORING_LUT_TILE_Q;

            __syncthreads();
            const float* tile_lut_src = d_lut + q_start * LUT_ENTRIES_PER_QUERY;
            for (int i = threadIdx.x; i < DOCUMENT_SCORING_LUT_SMEM_FLOATS; i += blockDim.x) {
                smem_lut[i] = tile_lut_src[i];
            }
            if (threadIdx.x < DOCUMENT_SCORING_LUT_TILE_Q) {
                smem_cb1_sumq[threadIdx.x] = d_cb1_sumq[q_start + threadIdx.x];
            }
            __syncthreads();

            if (valid) {
                #pragma unroll
                for (int tq = 0; tq < DOCUMENT_SCORING_LUT_TILE_Q; tq++) {
                    const float* q_lut = smem_lut + tq * LUT_ENTRIES_PER_QUERY;
                    float ip = 0.0f;

                    #pragma unroll
                    for (int c = 0; c < NUM_CHUNKS; c++) {
                        ip += q_lut[c * LUT_SIZE + nibbles[c]];
                    }

                    int q = q_start + tq;
                    d_one_bit_scores[q * total_tokens + tok_idx] =
                        (ip - smem_cb1_sumq[tq]) * factor;
                }
            }
        }
    }
}


__global__ void one_bit_document_score_kernel(
    const float*  __restrict__ d_one_bit_token_scores,
    const size_t* __restrict__ d_refined_doc_offsets,
    float*        d_one_bit_document_scores,
    size_t total_tokens,
    size_t num_refined_candidates
) {
    constexpr int TILE_T = 8;
    __shared__ float tile[Q_DOCLEN * TILE_T];
    __shared__ float max_vals[Q_DOCLEN];
    __shared__ float reduce_buf[256];

    size_t refined_idx = blockIdx.x;
    if (refined_idx >= num_refined_candidates) return;

    size_t tok_start = d_refined_doc_offsets[refined_idx];
    size_t tok_end = d_refined_doc_offsets[refined_idx + 1];
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
            tile[q * TILE_T + t_local] = d_one_bit_token_scores[q * total_tokens + tok_start + t_base + t_local];
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
        d_one_bit_document_scores[refined_idx] = reduce_buf[0];
    }
}




__global__ void gather_refined_token_positions_kernel(
    const int*    __restrict__ d_refined_doc_ids,
    const int*    __restrict__ d_doc_ptrs,
    const uint32_t* __restrict__ d_token_to_cluster_pos,
    const size_t* __restrict__ d_refined_doc_offsets,
    uint32_t*     d_refined_token_positions,
    size_t num_refined_candidates
) {
    size_t refined_idx = blockIdx.x;
    if (refined_idx >= num_refined_candidates) return;

    int doc_id = d_refined_doc_ids[refined_idx];
    int doc_start = d_doc_ptrs[doc_id];
    int doc_end = d_doc_ptrs[doc_id + 1];
    size_t out_offset = d_refined_doc_offsets[refined_idx];

    for (int t = threadIdx.x; t < (doc_end - doc_start); t += blockDim.x) {
        const size_t token_id = static_cast<size_t>(doc_start + t);
        d_refined_token_positions[out_offset + t] = d_token_to_cluster_pos[token_id];
    }
}




__global__ void gather_refined_doc_lengths_kernel(
    const int* __restrict__ d_refined_doc_ids,
    const int* __restrict__ d_doc_ptrs,
    int*       d_doc_lengths,
    size_t num_refined_candidates
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_refined_candidates) return;

    int doc_id = d_refined_doc_ids[idx];
    int doc_len = d_doc_ptrs[doc_id + 1] - d_doc_ptrs[doc_id];
    d_doc_lengths[idx] = doc_len;
}


}  // namespace Chimera
