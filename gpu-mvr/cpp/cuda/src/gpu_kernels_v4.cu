#include "gpu_kernels_v4.cuh"

#include <cfloat>

#ifdef GPU_MVR_COMPACT_DOC_BUFFER
// ---- Open-addressing hash table: doc_id → compact_slot ----
// Keys array is initialised to -1 (empty). Vals array to -1 (not yet assigned).
// Returns the compact_id assigned to doc_id (newly allocated or existing).
__device__ __forceinline__
int doc_ht_find_or_insert(
    int* __restrict__ ht_keys,         // [ht_capacity], init -1
    int* __restrict__ ht_vals,         // [ht_capacity], init -1
    int* __restrict__ touched_list,    // [max_compact_docs]
    int* __restrict__ num_inserted,    // scalar, init 0
    int  doc_id,
    int  max_compact_docs,
    unsigned int ht_mask               // ht_capacity - 1
) {
    unsigned int h = (unsigned int)doc_id * 2654435761u;   // Knuth multiplicative hash
    h &= ht_mask;

    const unsigned int ht_capacity = ht_mask + 1;
    for (unsigned int probe_count = 0; probe_count < ht_capacity; ++probe_count) {
        int probe = atomicCAS(&ht_keys[h], -1, doc_id);
        if (probe == -1) {
            // We just inserted this key. Allocate a compact id.
            int cid = atomicAdd(num_inserted, 1);
            if (cid >= max_compact_docs) {
                atomicExch(&ht_vals[h], -2);  // overflow sentinel
                return -1;
            }
            touched_list[cid] = doc_id;
            atomicExch(&ht_vals[h], cid);     // publish
            return cid;
        }
        if (probe == doc_id) {
            // Key already exists. Spin until compact_id is published.
            int cid;
            do { cid = *(volatile int*)&ht_vals[h]; } while (cid == -1);
            return (cid >= 0 && cid < max_compact_docs) ? cid : -1;
        }
        // Collision with a different key — linear probe.
        h = (h + 1) & ht_mask;
    }
    return -1;
}
#endif

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

#ifdef GPU_MVR_DOCID_VIA_DOCPTRS
__device__ __forceinline__ int doc_id_from_doc_ptrs_binary(
    const int* __restrict__ d_doc_ptrs,
    int num_docs,
    uint32_t token_id
) {
    int lo = 0;
    int hi = num_docs - 1;

    while (lo < hi) {
        const int mid = lo + ((hi - lo + 1) >> 1);
        const uint32_t mid_start = (uint32_t)__ldg(&d_doc_ptrs[mid]);
        if (mid_start <= token_id) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    return lo;
}

__device__ __forceinline__ int doc_id_from_doc_ptrs_binary_range(
    const int* __restrict__ d_doc_ptrs,
    int lo,
    int hi,
    uint32_t token_id
) {
    while (lo < hi) {
        const int mid = lo + ((hi - lo + 1) >> 1);
        const uint32_t mid_start = (uint32_t)__ldg(&d_doc_ptrs[mid]);
        if (mid_start <= token_id) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    return lo;
}

__device__ __forceinline__ int doc_id_from_doc_ptrs_blocked(
    const int* __restrict__ d_doc_ptrs,
    const int* __restrict__ d_doc_block_lut,
    uint32_t token_id
) {
    const int block = static_cast<int>(token_id >> kDocPtrLookupBlockShift);
    const int lo = __ldg(&d_doc_block_lut[block]);
    const int hi = __ldg(&d_doc_block_lut[block + 1]);
    return (lo >= hi)
        ? lo
        : doc_id_from_doc_ptrs_binary_range(d_doc_ptrs, lo, hi, token_id);
}

__device__ __forceinline__ bool warp_tokens_are_nondecreasing(
    uint32_t token_id,
    unsigned active_mask
) {
    const int lane = threadIdx.x & 31;
    const unsigned lower_lanes_mask =
        active_mask & ((lane == 0) ? 0u : ((1u << lane) - 1u));
    bool ordered = true;
    if (lower_lanes_mask != 0u) {
        const int prev_lane = 31 - __clz(lower_lanes_mask);
        const uint32_t prev_token = __shfl_sync(active_mask, token_id, prev_lane);
        ordered = token_id >= prev_token;
    }
    return __all_sync(active_mask, ordered);
}

__device__ __forceinline__ int doc_id_from_doc_ptrs_warp_sorted(
    uint32_t token_id,
    const int* __restrict__ d_doc_ptrs,
    int num_docs,
    unsigned active_mask,
    int* __restrict__ warp_doc_ids
) {
    const int lane = threadIdx.x & 31;
    const int leader_lane = __ffs(active_mask) - 1;

    if (lane == leader_lane) {
        unsigned remaining = active_mask;
        int lower_doc = 0;
        bool first = true;

        while (remaining != 0u) {
            const int src_lane = __ffs(remaining) - 1;
            const uint32_t lane_token = __shfl_sync(active_mask, token_id, src_lane);
            lower_doc = first
                ? doc_id_from_doc_ptrs_binary(d_doc_ptrs, num_docs, lane_token)
                : doc_id_from_doc_ptrs_binary_range(
                    d_doc_ptrs,
                    lower_doc,
                    num_docs - 1,
                    lane_token);
            warp_doc_ids[src_lane] = lower_doc;
            remaining &= (remaining - 1);
            first = false;
        }
    }

    __syncwarp(active_mask);
    return warp_doc_ids[lane];
}
#endif

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

// Fused stage1 kernel: computes per-(query, emb) binary IP distances and
// directly performs the doc-level atomicMax aggregation + touched-list
// tracking inline.
//
// v3 — flat iteration with preloaded cluster metadata:
//   1. Cooperatively preload all (cluster_start, cluster_size) pairs
//      into shared memory, eliminating per-cluster global loads from
//      the hot loop.
//   2. Compute a prefix sum of cluster sizes → flat element indices.
//   3. Single grid-stride loop over ALL elements across all clusters.
//      A 7-step binary search (on smem) maps flat_idx → (cluster, local).
//      This removes the 128-iteration cluster loop overhead and ensures
//      every thread does useful work (no idle threads on small clusters).
//   4. 128-bit vectorized code load (uint4) + 32-bit nibble extraction
//      (halves shift instructions vs uint64_t).
//   5. Read-before-atomic guard on d_doc_query_max.
//
// Grid: (blocks_x, Q_DOCLEN).
// Requires d_doc_query_max to be zero-initialised.
#define STAGE1_MAX_NPROBE 512

__global__ void stage1_binary_ip_lut_kernel(
    const float*    __restrict__ d_lut,
    const char*     __restrict__ d_clustered_code,
    const float*    __restrict__ d_clustered_factor,
    const float*    __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    float*       d_doc_query_max,
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
    int*         d_ht_keys,
    int*         d_ht_vals,
    int*         d_touched_doc_list,
    int*         d_num_touched_docs,
    int          max_touched_docs,
    unsigned int ht_mask,
#else
    int*         d_doc_touched,
    int*         d_touched_doc_list,
    int*         d_num_touched_docs,
#endif
    size_t       num_docs,
    int          nprobe,
    size_t       n_clusters
) {
    __shared__ float    smem_lut[LUT_ENTRIES_PER_QUERY];     // 2048 B
    __shared__ uint32_t smem_cstart[STAGE1_MAX_NPROBE];      // 1024 B
    __shared__ int      smem_prefix[STAGE1_MAX_NPROBE + 1];  // 1028 B

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) return;

    // ---- Phase 1: Load LUT into smem ----
    const float* lut_ptr = d_lut + query_idx * LUT_ENTRIES_PER_QUERY;
    for (int i = threadIdx.x; i < LUT_ENTRIES_PER_QUERY; i += blockDim.x) {
        smem_lut[i] = lut_ptr[i];
    }

    // ---- Phase 2: Cooperatively load cluster metadata ----
    const uint32_t* my_labels = d_cagra_labels + query_idx * nprobe;
    for (int p = threadIdx.x; p < nprobe; p += blockDim.x) {
        uint32_t label = my_labels[p];
        if (label < (uint32_t)n_clusters) {
            size_t start = d_cluster_pos[label];
            smem_cstart[p] = (uint32_t)start;
            smem_prefix[p + 1] = (int)(d_cluster_pos[label + 1] - start);
        } else {
            smem_cstart[p] = 0;
            smem_prefix[p + 1] = 0;
        }
    }
    if (threadIdx.x == 0) smem_prefix[0] = 0;
    __syncthreads();

    // ---- Phase 3: Prefix sum of cluster sizes (sequential, ~128 adds) ----
    if (threadIdx.x == 0) {
        for (int i = 1; i <= nprobe; i++)
            smem_prefix[i] += smem_prefix[i - 1];
    }
    __syncthreads();

    const int   total_elements = smem_prefix[nprobe];
    const float cb1_sumq       = d_cb1_sumq[query_idx];

    // ---- Phase 4: Flat grid-stride loop over all elements ----
    for (int flat_idx = threadIdx.x + blockIdx.x * blockDim.x;
         flat_idx < total_elements;
         flat_idx += blockDim.x * gridDim.x)
    {
        // Binary search: find the probe whose prefix range contains flat_idx.
        // 7 steps for nprobe ≤ 128, all on shared memory (~28 cycles).
        int lo = 0, hi = nprobe;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (smem_prefix[mid + 1] <= flat_idx)
                lo = mid + 1;
            else
                hi = mid;
        }
        uint32_t emb_pos = smem_cstart[lo] + (flat_idx - smem_prefix[lo]);

        // 128-bit vectorized code load (one LDG.128 instead of two LDG.64).
        const uint4 code128 = *reinterpret_cast<const uint4*>(
            d_clustered_code + static_cast<size_t>(emb_pos) * CODE_BYTES);

        // Issue factor + doc_id loads early — their ~400-cycle latency
        // overlaps with the 32-step LUT computation below.
        const float factor = d_clustered_factor[emb_pos];
        const int   doc_id = d_clustered_doc_ids[emb_pos];

        // 32-bit nibble extraction: uses native 32-bit shifts (1 PTX insn
        // each) instead of 64-bit shifts (2 PTX insns each). 4 unrolled
        // blocks of 8 nibbles from uint4 members .x/.y/.z/.w.
        float ip = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; n++)
            ip += smem_lut[n * LUT_SIZE + ((code128.x >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; n++)
            ip += smem_lut[(8 + n) * LUT_SIZE + ((code128.y >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; n++)
            ip += smem_lut[(16 + n) * LUT_SIZE + ((code128.z >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; n++)
            ip += smem_lut[(24 + n) * LUT_SIZE + ((code128.w >> (n * 4)) & 0xF)];

        float dist = (ip - cb1_sumq) * factor;

        if (dist > 0.0f && (size_t)doc_id < num_docs) {
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
            int cid = doc_ht_find_or_insert(
                d_ht_keys, d_ht_vals,
                d_touched_doc_list, d_num_touched_docs,
                doc_id, max_touched_docs, ht_mask);
            if (cid < 0) {
                continue;
            }

            float* slot = &d_doc_query_max[(size_t)cid * Q_DOCLEN + query_idx];
#else
            float* slot = &d_doc_query_max[(size_t)doc_id * Q_DOCLEN + query_idx];
#endif
            if (dist > *slot) {
                atomicMax(
                    reinterpret_cast<int*>(slot),
                    __float_as_int(dist));
            }

#ifndef GPU_MVR_COMPACT_DOC_BUFFER
            if (d_doc_touched[doc_id] == 0) {
                int was_touched = atomicExch(&d_doc_touched[doc_id], 1);
                if (was_touched == 0) {
                    int pos = atomicAdd(d_num_touched_docs, 1);
                    d_touched_doc_list[pos] = doc_id;
                }
            }
#endif
        }
    }
}

// ---------------------------------------------------------------------------
// Non-clustered variant: same flat grid-stride structure as the clustered
// kernel to preserve full parallelism, but uses inv_list indirection and
// __ldg() for scattered code/factor/doc_id reads through the read-only cache.
// ---------------------------------------------------------------------------

__global__ void stage1_binary_ip_lut_nonclustered_kernel(
    const float*    __restrict__ d_lut,
    const char*     __restrict__ d_code,
    const float*    __restrict__ d_factor,
    const float*    __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
#ifdef GPU_MVR_DOCID_VIA_DOCPTRS
    const int*      __restrict__ d_doc_ptrs,
    const int*      __restrict__ d_doc_block_lut,
#else
    const int*      __restrict__ d_doc_ids,
#endif
    const int*      __restrict__ d_inv_list,
    float*       d_doc_query_max,
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
    int*         d_ht_keys,
    int*         d_ht_vals,
    int*         d_touched_doc_list,
    int*         d_num_touched_docs,
    int          max_touched_docs,
    unsigned int ht_mask,
#else
    int*         d_doc_touched,
    int*         d_touched_doc_list,
    int*         d_num_touched_docs,
#endif
    size_t       num_docs,
    int          nprobe,
    size_t       n_clusters
) {
    __shared__ float    smem_lut[LUT_ENTRIES_PER_QUERY];     // 2048 B
    __shared__ uint32_t smem_cstart[STAGE1_MAX_NPROBE];      // 1024 B
    __shared__ int      smem_prefix[STAGE1_MAX_NPROBE + 1];  // 1028 B

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) return;

    // ---- Phase 1: Load LUT into smem ----
    const float* lut_ptr = d_lut + query_idx * LUT_ENTRIES_PER_QUERY;
    for (int i = threadIdx.x; i < LUT_ENTRIES_PER_QUERY; i += blockDim.x) {
        smem_lut[i] = lut_ptr[i];
    }

    // ---- Phase 2: Cooperatively load cluster metadata ----
    const uint32_t* my_labels = d_cagra_labels + query_idx * nprobe;
    for (int p = threadIdx.x; p < nprobe; p += blockDim.x) {
        uint32_t label = my_labels[p];
        if (label < (uint32_t)n_clusters) {
            size_t start = d_cluster_pos[label];
            smem_cstart[p] = (uint32_t)start;
            smem_prefix[p + 1] = (int)(d_cluster_pos[label + 1] - start);
        } else {
            smem_cstart[p] = 0;
            smem_prefix[p + 1] = 0;
        }
    }
    if (threadIdx.x == 0) smem_prefix[0] = 0;
    __syncthreads();

    // ---- Phase 3: Prefix sum of cluster sizes ----
    if (threadIdx.x == 0) {
        for (int i = 1; i <= nprobe; i++)
            smem_prefix[i] += smem_prefix[i - 1];
    }
    __syncthreads();

    const int   total_elements = smem_prefix[nprobe];
    const float cb1_sumq       = d_cb1_sumq[query_idx];

    // ---- Phase 4: Flat grid-stride loop over all elements ----
    for (int flat_idx = threadIdx.x + blockIdx.x * blockDim.x;
         flat_idx < total_elements;
         flat_idx += blockDim.x * gridDim.x)
    {
        // Binary search for the cluster containing flat_idx
        int lo = 0, hi = nprobe;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (smem_prefix[mid + 1] <= flat_idx)
                lo = mid + 1;
            else
                hi = mid;
        }
        uint32_t emb_pos = smem_cstart[lo] + (flat_idx - smem_prefix[lo]);

        // inv_list indirection: map cluster position -> original vector ID
        uint32_t orig_id = (uint32_t)__ldg(&d_inv_list[emb_pos]);

        // __ldg() routes scattered reads through the read-only texture cache
        const uint4 code128 = __ldg(reinterpret_cast<const uint4*>(
            d_code + (size_t)orig_id * CODE_BYTES));
        const float factor = __ldg(&d_factor[orig_id]);
#ifdef GPU_MVR_DOCID_VIA_DOCPTRS
        const int doc_id = doc_id_from_doc_ptrs_blocked(
            d_doc_ptrs,
            d_doc_block_lut,
            orig_id);
#else
        const int   doc_id = __ldg(&d_doc_ids[orig_id]);
#endif

        float ip = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; n++)
            ip += smem_lut[n * LUT_SIZE + ((code128.x >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; n++)
            ip += smem_lut[(8 + n) * LUT_SIZE + ((code128.y >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; n++)
            ip += smem_lut[(16 + n) * LUT_SIZE + ((code128.z >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; n++)
            ip += smem_lut[(24 + n) * LUT_SIZE + ((code128.w >> (n * 4)) & 0xF)];

        float dist = (ip - cb1_sumq) * factor;

        if (dist > 0.0f && (size_t)doc_id < num_docs) {
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
            int cid = doc_ht_find_or_insert(
                d_ht_keys, d_ht_vals,
                d_touched_doc_list, d_num_touched_docs,
                doc_id, max_touched_docs, ht_mask);
            if (cid < 0) {
                continue;
            }

            float* slot = &d_doc_query_max[(size_t)cid * Q_DOCLEN + query_idx];
#else
            float* slot = &d_doc_query_max[(size_t)doc_id * Q_DOCLEN + query_idx];
#endif
            if (dist > *slot) {
                atomicMax(
                    reinterpret_cast<int*>(slot),
                    __float_as_int(dist));
            }

#ifndef GPU_MVR_COMPACT_DOC_BUFFER
            if (d_doc_touched[doc_id] == 0) {
                int was_touched = atomicExch(&d_doc_touched[doc_id], 1);
                if (was_touched == 0) {
                    int pos = atomicAdd(d_num_touched_docs, 1);
                    d_touched_doc_list[pos] = doc_id;
                }
            }
#endif
        }
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
    size_t              num_queries,
    bool                use_clustered_layout
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

        if (use_clustered_layout) {
            // Emit the cluster-local position directly. The caller will
            // index into d_clustered_code/factor/doc_ids which are already
            // reordered by inv_list, so the position IS the access index.
            for (size_t i = threadIdx.x; i < cluster_size; i += blockDim.x) {
                d_emb_ids[out_start + write_pos + i] = cluster_start + i;
            }
        } else {
            for (size_t i = threadIdx.x; i < cluster_size; i += blockDim.x) {
                d_emb_ids[out_start + write_pos + i] = d_inv_list[cluster_start + i];
            }
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

__global__ void aggregate_stage1_tracked_kernel(
    const size_t* __restrict__ d_emb_ids,
    const float*  __restrict__ d_emb_dists,
    const int*    __restrict__ d_pair_offsets,
#ifdef GPU_MVR_DOCID_VIA_DOCPTRS
    const int*    __restrict__ d_doc_ptrs,
    const int*    __restrict__ d_doc_block_lut,
#else
    const int*    __restrict__ d_doc_ids,
#endif
    float*        d_doc_query_max,
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
    int*          d_ht_keys,
    int*          d_ht_vals,
    int*          d_touched_doc_list,
    int*          d_num_touched_docs,
    int           max_touched_docs,
    unsigned int  ht_mask,
#else
    int*          d_doc_touched,
    int*          d_touched_doc_list,
    int*          d_num_touched_docs,
#endif
    size_t        num_docs,
    size_t        total_pairs,
    size_t        max_embs_per_query
) {
    (void)total_pairs;

    // Grid: (blocks_x, Q_DOCLEN). blockIdx.y encodes q_idx, eliminating binary search.
    const int q_idx = blockIdx.y;
    if (q_idx >= Q_DOCLEN) return;

    const int pair_start = d_pair_offsets[q_idx];
    const int pair_end   = d_pair_offsets[q_idx + 1];
    const int num_embs   = pair_end - pair_start;

    const int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= num_embs) return;

    // Load emb dist first. If it is non-positive, the atomicMax on the
    // zero-initialized d_doc_query_max is guaranteed to be a no-op, so we
    // can skip both the atomic and the touched-list bookkeeping. This is
    // the single biggest cost in this kernel — in the doc-major layout
    // each atomic goes to its own cache line, so halving the atomic count
    // ~halves the kernel time. A doc whose contributions are all <= 0
    // would have score 0 and can never appear in top-k anyway, so dropping
    // it from the touched list is safe.
    const float dist =
        d_emb_dists[(size_t)q_idx * max_embs_per_query + local_idx];

    const size_t emb_id = d_emb_ids[(size_t)pair_start + local_idx];
#ifdef GPU_MVR_DOCID_VIA_DOCPTRS
    const uint32_t token_id = static_cast<uint32_t>(emb_id);
    const int doc_id = doc_id_from_doc_ptrs_blocked(
        d_doc_ptrs,
        d_doc_block_lut,
        token_id);
#else
    const int doc_id = d_doc_ids[emb_id];
#endif

    if (dist <= 0.0f) return;
    if (doc_id >= (int)num_docs) return;

#ifdef GPU_MVR_COMPACT_DOC_BUFFER
    // Doc-major layout: [compact_id * Q_DOCLEN + q_idx]. Adjacent q's for
    // same doc are contiguous, which makes the subsequent sum kernel coalesced.
    int cid = doc_ht_find_or_insert(
        d_ht_keys, d_ht_vals,
        d_touched_doc_list, d_num_touched_docs,
        doc_id, max_touched_docs, ht_mask);
    if (cid < 0) return;
    atomicMaxFloat(&d_doc_query_max[(size_t)cid * Q_DOCLEN + q_idx], dist);
#else
    // Doc-major layout: [doc_id * Q_DOCLEN + q_idx]. Adjacent q's for same
    // doc are contiguous, which makes the subsequent sum kernel coalesced.
    atomicMaxFloat(&d_doc_query_max[(size_t)doc_id * Q_DOCLEN + q_idx], dist);

    if (d_doc_touched[doc_id] == 0) {
        int was_touched = atomicExch(&d_doc_touched[doc_id], 1);
        if (was_touched == 0) {
            int pos = atomicAdd(d_num_touched_docs, 1);
            d_touched_doc_list[pos] = doc_id;
        }
    }
#endif
}

// Warp-per-doc cooperative summation.
// Q_DOCLEN == 32 == warp size, so each lane loads exactly one float,
// then we warp-shuffle reduce.  This replaces 8 serial float4 loads
// per thread with a single float load + 5-step shuffle — a dramatic
// reduction in per-doc L2 traffic and latency.
// Launch: <<<(num_touched + WARPS_PER_BLOCK-1)/WARPS_PER_BLOCK, WARPS_PER_BLOCK*32>>>

__global__ void sum_doc_scores_sparse_kernel(
    const float* __restrict__ d_doc_query_max,
    const int*   __restrict__ d_touched_doc_list,
    float*       d_scores_out,
    int*         d_doc_ids_out,
    int          num_touched
#ifndef GPU_MVR_COMPACT_DOC_BUFFER
    , size_t     num_docs
#endif
) {
    const int warp_id_in_block = threadIdx.x >> 5;          // 0..7
    const int lane             = threadIdx.x & 31;          // 0..31
    const int global_warp_id   = blockIdx.x * SUM_SCORES_WARPS_PER_BLOCK + warp_id_in_block;

    if (global_warp_id >= num_touched) return;

    const int doc_id = d_touched_doc_list[global_warp_id];

#ifdef GPU_MVR_COMPACT_DOC_BUFFER
    // d_doc_query_max is compact-indexed: row = global_warp_id
    // (== compact_id because touched_list is in compact_id order).
    float val = d_doc_query_max[(size_t)global_warp_id * Q_DOCLEN + lane];
#else
    // d_doc_query_max is indexed by doc_id directly.
    float val = d_doc_query_max[(size_t)doc_id * Q_DOCLEN + lane];
#endif

    // Warp-shuffle tree reduction (5 steps for 32 lanes).
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    if (lane == 0) {
        d_scores_out[global_warp_id]  = val;
        d_doc_ids_out[global_warp_id] = doc_id;
    }
}
