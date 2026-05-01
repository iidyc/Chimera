#include "gpu_kernels.cuh"

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

__device__ __forceinline__ void bitmap_flag_doc(
    doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    int doc_id
) {
    const int bucket = doc_id / kDocBitmapBucketWidth;
    const int bit = doc_id % kDocBitmapBucketWidth;
    atomicOr(&d_doc_bitmap[bucket], doc_bitmap_bucket_t(1u) << bit);
}

__device__ __forceinline__ int bitmap_compact_id(
    const doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    const doc_bitmap_offset_t* __restrict__ d_doc_bitmap_offsets,
    int doc_id
) {
    const int bucket = doc_id / kDocBitmapBucketWidth;
    const int group = bucket / kDocBitmapCompRatio;
    const int group_bucket_start = group * kDocBitmapCompRatio;

    int compact_id = static_cast<int>(d_doc_bitmap_offsets[group]);
    #pragma unroll
    for (int b = 0; b < kDocBitmapCompRatio; ++b) {
        const int bucket_idx = group_bucket_start + b;
        if (bucket_idx >= bucket) {
            break;
        }
        compact_id += __popc(__ldg(&d_doc_bitmap[bucket_idx]));
    }

    const int bit = doc_id % kDocBitmapBucketWidth;
    const doc_bitmap_bucket_t bucket_bits = __ldg(&d_doc_bitmap[bucket]);
    const doc_bitmap_bucket_t prior_bits =
        (bit == 0)
            ? doc_bitmap_bucket_t(0)
            : (bucket_bits & ((doc_bitmap_bucket_t(1u) << bit) - 1u));
    compact_id += __popc(prior_bits);
    return compact_id;
}

__device__ __forceinline__ uint32_t cluster_virtual_to_actual_emb_pos(
    uint32_t cluster_start,
    int cluster_size,
    int virtual_local_idx
) {
    const int misalign = static_cast<int>(cluster_start & (kStage1ClusterAlignTokens - 1));
    const int head = (misalign == 0)
        ? 0
        : static_cast<int>(kStage1ClusterAlignTokens) - misalign;
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

__global__ void bitmap_offset_init_kernel(
    const doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    size_t num_buckets,
    doc_bitmap_offset_t* __restrict__ d_doc_bitmap_offsets
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_buckets) {
        return;
    }

    if ((idx % kDocBitmapCompRatio) == 0) {
        doc_bitmap_offset_t count = 0;
        const size_t group_bucket_start = idx;
        #pragma unroll
        for (int offset = 0; offset < kDocBitmapCompRatio; ++offset) {
            const size_t bucket_idx = group_bucket_start + offset;
            if (bucket_idx >= num_buckets) {
                break;
            }
            count += __popc(d_doc_bitmap[bucket_idx]);
        }
        d_doc_bitmap_offsets[idx / kDocBitmapCompRatio] = count;
    }
}

__global__ void bitmap_unique_docs_kernel(
    const doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    size_t num_buckets,
    const doc_bitmap_offset_t* __restrict__ d_doc_bitmap_offsets,
    int* __restrict__ d_out_doc_ids
) {
    const size_t bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= num_buckets) {
        return;
    }

    const size_t group_idx = bucket_idx / kDocBitmapCompRatio;
    const size_t group_bucket_start = group_idx * kDocBitmapCompRatio;
    uint32_t out_idx = d_doc_bitmap_offsets[group_idx];

    #pragma unroll
    for (int offset = 0; offset < kDocBitmapCompRatio; ++offset) {
        const size_t prev_bucket = group_bucket_start + offset;
        if (prev_bucket >= bucket_idx) {
            break;
        }
        out_idx += __popc(__ldg(&d_doc_bitmap[prev_bucket]));
    }

    doc_bitmap_bucket_t bits = __ldg(&d_doc_bitmap[bucket_idx]);
    while (bits != 0u) {
        const int bit = __ffs(bits) - 1;
        d_out_doc_ids[out_idx++] =
            static_cast<int>(bucket_idx * kDocBitmapBucketWidth + bit);
        bits &= (bits - 1);
    }
}

__global__ void sum_doc_scores_compact_kernel(
    const float* __restrict__ d_doc_query_max,
    const int*   __restrict__ d_touched_doc_list,
    float*       d_scores_out,
    int*         d_doc_ids_out,
    int          num_touched
) {
    const int warp_id_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int global_warp_id =
        blockIdx.x * SUM_SCORES_WARPS_PER_BLOCK + warp_id_in_block;

    if (global_warp_id >= num_touched) {
        return;
    }

    const int doc_id = d_touched_doc_list[global_warp_id];
    float val =
        d_doc_query_max[static_cast<size_t>(global_warp_id) * Q_DOCLEN + lane];

    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (lane == 0) {
        d_scores_out[global_warp_id] = val;
        d_doc_ids_out[global_warp_id] = doc_id;
    }
}

__global__ void stage1_binary_ip_lut_flag_docs_kernel(
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    size_t          num_docs,
    int             nprobe,
    size_t          n_clusters
) {
    __shared__ uint32_t smem_cstart[kStage1MaxNprobe];
    __shared__ int smem_prefix[kStage1MaxNprobe + 1];

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
        const uint32_t emb_pos = cluster_virtual_to_actual_emb_pos(
            smem_cstart[lo], cluster_size, cluster_local);
        const int doc_id = d_clustered_doc_ids[emb_pos];

        if (static_cast<size_t>(doc_id) < num_docs) {
            const unsigned int active_mask = __activemask();
            const unsigned int doc_group = warp_matching_doc_mask(doc_id, active_mask);
            const int lane = threadIdx.x & 31;
            if (lane == (__ffs(doc_group) - 1)) {
                bitmap_flag_doc(d_doc_bitmap, doc_id);
            }
        }
    }
}

__global__ void stage1_binary_ip_lut_compact_kernel(
    const float*    __restrict__ d_lut,
    const char*     __restrict__ d_clustered_code,
    const float*    __restrict__ d_clustered_factor,
    const float*    __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    float*          __restrict__ d_doc_query_max,
    const doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    const doc_bitmap_offset_t* __restrict__ d_doc_bitmap_offsets,
    int             max_touched_docs,
    size_t          num_docs,
    int             nprobe,
    size_t          n_clusters
) {
    __shared__ float smem_lut[LUT_ENTRIES_PER_QUERY];
    __shared__ uint32_t smem_cstart[kStage1MaxNprobe];
    __shared__ int smem_prefix[kStage1MaxNprobe + 1];

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
        const uint32_t emb_pos = cluster_virtual_to_actual_emb_pos(
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
                const int cid = bitmap_compact_id(
                    d_doc_bitmap,
                    d_doc_bitmap_offsets,
                    doc_id);
                if (cid < 0 || cid >= max_touched_docs) {
                    continue;
                }
                atomicMaxFloat(
                    &d_doc_query_max[static_cast<size_t>(cid) * Q_DOCLEN + query_idx],
                    group_max);
            }
        }
    }
}

__global__ void stage1_binary_ip_nolut_compact_kernel(
    const float*    __restrict__ d_queries,
    const char*     __restrict__ d_clustered_code,
    const float*    __restrict__ d_clustered_factor,
    const float*    __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    float*          __restrict__ d_doc_query_max,
    const doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    const doc_bitmap_offset_t* __restrict__ d_doc_bitmap_offsets,
    int             max_touched_docs,
    size_t          num_docs,
    int             nprobe,
    size_t          n_clusters
) {
    __shared__ float smem_query[PADDED_DIM];
    __shared__ uint32_t smem_cstart[kStage1MaxNprobe];
    __shared__ int smem_prefix[kStage1MaxNprobe + 1];

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) {
        return;
    }

    const float* q_ptr = d_queries + query_idx * PADDED_DIM;
    for (int i = threadIdx.x; i < PADDED_DIM; i += blockDim.x) {
        smem_query[i] = q_ptr[i];
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
        const uint32_t emb_pos = cluster_virtual_to_actual_emb_pos(
            smem_cstart[lo], cluster_size, cluster_local);

        const uint64_t* code_ptr = reinterpret_cast<const uint64_t*>(
            d_clustered_code + static_cast<size_t>(emb_pos) * CODE_BYTES);
        const uint64_t code0 = code_ptr[0];
        const uint64_t code1 = code_ptr[1];
        const float factor = d_clustered_factor[emb_pos];
        const int doc_id = d_clustered_doc_ids[emb_pos];

        float ip = 0.0f;
        uint64_t bits = code0;
        while (bits) {
            const int pos = __ffsll(bits) - 1;
            ip += smem_query[pos];
            bits &= bits - 1;
        }
        bits = code1;
        while (bits) {
            const int pos = __ffsll(bits) - 1;
            ip += smem_query[64 + pos];
            bits &= bits - 1;
        }

        const float dist = (ip - cb1_sumq) * factor;
        if (dist > 0.0f && static_cast<size_t>(doc_id) < num_docs) {
            const unsigned int active_mask = __activemask();
            const unsigned int doc_group = warp_matching_doc_mask(doc_id, active_mask);
            const int lane = threadIdx.x & 31;
            const int leader_lane = __ffs(doc_group) - 1;
            const float group_max = warp_group_max(dist, doc_group, active_mask);

            if (lane == leader_lane) {
                const int cid = bitmap_compact_id(
                    d_doc_bitmap,
                    d_doc_bitmap_offsets,
                    doc_id);
                if (cid < 0 || cid >= max_touched_docs) {
                    continue;
                }
                atomicMaxFloat(
                    &d_doc_query_max[static_cast<size_t>(cid) * Q_DOCLEN + query_idx],
                    group_max);
            }
        }
    }
}


}  // namespace Chimera
