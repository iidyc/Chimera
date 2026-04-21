#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "gpu_config.cuh"

// Query-tiled stage2 LUT kernel constants (used at launch sites for shared mem sizing)
#define STAGE2_LUT_TILE_Q 8
#define STAGE2_LUT_NUM_TILES (Q_DOCLEN / STAGE2_LUT_TILE_Q)
#define STAGE2_LUT_SMEM_FLOATS (STAGE2_LUT_TILE_Q * LUT_ENTRIES_PER_QUERY)

#ifdef GPU_MVR_COMPACT_DOC_BUFFER
using doc_bitmap_bucket_t = uint32_t;
using doc_bitmap_offset_t = uint32_t;

inline constexpr int kDocBitmapBucketWidth = 32;
inline constexpr doc_bitmap_bucket_t kDocBitmapHighestBit = 0x80000000u;
inline constexpr int kDocBitmapCompRatio = 8;

inline constexpr size_t doc_bitmap_num_buckets(size_t num_docs) {
    const size_t raw =
        (num_docs + kDocBitmapBucketWidth - 1) / kDocBitmapBucketWidth;
    return ((raw + kDocBitmapCompRatio - 1) / kDocBitmapCompRatio) *
           kDocBitmapCompRatio;
}

inline constexpr size_t doc_bitmap_num_offsets(size_t num_buckets) {
    return (num_buckets / kDocBitmapCompRatio) + 1;
}
#endif

#ifdef GPU_MVR_V6_GPU_INDICES_U32
using stage2_token_id_t = uint32_t;
using gpu_cluster_pos_t = uint32_t;
#else
using stage2_token_id_t = size_t;
using gpu_cluster_pos_t = size_t;
#endif

inline constexpr int kDocPtrLookupBlockShift = 9;
inline constexpr uint32_t kDocPtrLookupBlockSize =
    1u << kDocPtrLookupBlockShift;

__global__ void stage1_binary_ip_kernel_v2(
    const float* __restrict__ d_queries,
    const char*  __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_emb_ids,
    const int*   __restrict__ d_pair_offsets,
    float* __restrict__ d_out_dists,
    size_t max_embs_per_query
);

__global__ void stage2_binary_ip_kernel_v2(
    const float* __restrict__ d_queries,
    const char*  __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const stage2_token_id_t* __restrict__ d_token_ids,
    float* __restrict__ d_out_dists,
    size_t total_tokens,
    size_t batch_tokens
);

__global__ void precompute_lut_kernel(
    const float* __restrict__ d_queries,
    float* __restrict__ d_lut
);

__global__ void stage1_binary_ip_lut_kernel(
    const float*    __restrict__ d_lut,
    const char*     __restrict__ d_clustered_code,
    const float*    __restrict__ d_clustered_factor,
    const float*    __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_cagra_labels,
    const gpu_cluster_pos_t* __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    float*       d_doc_query_max,
    const doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    const doc_bitmap_offset_t* __restrict__ d_doc_bitmap_offsets,
    int          max_touched_docs,
    size_t       num_docs,
    int          nprobe,
    size_t       n_clusters
);

__global__ void stage1_binary_ip_lut_flag_docs_kernel(
    const uint32_t* __restrict__ d_cagra_labels,
    const gpu_cluster_pos_t* __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    doc_bitmap_bucket_t* d_doc_bitmap,
    size_t       num_docs,
    int          nprobe,
    size_t       n_clusters
);

// Non-clustered variant: uses inv_list indirection to access original arrays.
// Optimized for scattered access with explicit cluster iteration, shared-memory
// inv_list tiling, and __ldg() for read-only cache utilization.
__global__ void stage1_binary_ip_lut_nonclustered_kernel(
    const float*    __restrict__ d_lut,
    const char*     __restrict__ d_code,
    const float*    __restrict__ d_factor,
    const float*    __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_cagra_labels,
    const gpu_cluster_pos_t* __restrict__ d_cluster_pos,
    const int*      __restrict__ d_doc_ids,
    const int*      __restrict__ d_inv_list,
    float*       d_doc_query_max,
    const doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    const doc_bitmap_offset_t* __restrict__ d_doc_bitmap_offsets,
    int          max_touched_docs,
    size_t       num_docs,
    int          nprobe,
    size_t       n_clusters
);

__global__ void stage1_binary_ip_lut_nonclustered_docptr_kernel(
    const float*    __restrict__ d_lut,
    const char*     __restrict__ d_code,
    const float*    __restrict__ d_factor,
    const float*    __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_cagra_labels,
    const gpu_cluster_pos_t* __restrict__ d_cluster_pos,
    const int*      __restrict__ d_doc_ptrs,
    const int*      __restrict__ d_doc_block_lut,
    const int*      __restrict__ d_inv_list,
    float*       d_doc_query_max,
    const doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    const doc_bitmap_offset_t* __restrict__ d_doc_bitmap_offsets,
    int          max_touched_docs,
    size_t       num_docs,
    int          nprobe,
    size_t       n_clusters
);

__global__ void stage1_binary_ip_lut_nonclustered_flag_docs_kernel(
    const uint32_t* __restrict__ d_cagra_labels,
    const gpu_cluster_pos_t* __restrict__ d_cluster_pos,
    const int*      __restrict__ d_doc_ids,
    const int*      __restrict__ d_doc_ptrs,
    const int*      __restrict__ d_doc_block_lut,
    const int*      __restrict__ d_inv_list,
    doc_bitmap_bucket_t* d_doc_bitmap,
    size_t       num_docs,
    int          nprobe,
    size_t       n_clusters
);

__global__ void bitmap_offset_init_kernel(
    const doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    size_t num_buckets,
    doc_bitmap_offset_t* __restrict__ d_doc_bitmap_offsets
);

__global__ void bitmap_unique_docs_kernel(
    const doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    size_t num_buckets,
    const doc_bitmap_offset_t* __restrict__ d_doc_bitmap_offsets,
    int* __restrict__ d_out_doc_ids
);

__global__ void stage2_binary_ip_lut_kernel(
    const float* __restrict__ d_lut,
    const char*  __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const stage2_token_id_t* __restrict__ d_token_ids,
    float* __restrict__ d_out_dists,
    size_t total_tokens,
    size_t batch_tokens
);

__global__ void doc_score_kernel(
    const float*  __restrict__ d_token_dists,
    const size_t* __restrict__ d_candidate_offsets,
    float*        d_doc_scores,
    size_t total_tokens,
    size_t num_candidates
);

__global__ void extract_one_bit_dists_kernel(
    const float*  __restrict__ d_token_dists,
    const size_t* __restrict__ d_candidate_offsets,
    const int*    __restrict__ d_selected_indices,
    float*        d_out_one_bit_dists,
    const size_t* __restrict__ d_out_offsets,
    size_t total_tokens,
    size_t k
);

__global__ void extract_one_bit_dists_kernel_v2(
    const float*  __restrict__ d_token_dists,
    const size_t* __restrict__ d_candidate_offsets,
    const int*    __restrict__ d_selected_indices,
    float*        h_pinned_out,
    const size_t* __restrict__ d_out_offsets,
    size_t total_tokens,
    size_t k
);

__global__ void gather_token_ids_kernel(
    const int*    __restrict__ d_candidate_doc_ids,
    const int*    __restrict__ d_doc_ptrs,
    const size_t* __restrict__ d_candidate_offsets,
    stage2_token_id_t* d_out_token_ids,
    size_t num_candidates
);

__global__ void compute_query_expansion_sizes_kernel(
    const uint32_t* __restrict__ d_cagra_labels,
    const gpu_cluster_pos_t* __restrict__ d_cluster_pos,
    int*                d_query_sizes,
    int                 nprobe,
    size_t              n_clusters,
    size_t              num_queries
);

__global__ void expand_cluster_ids_kernel(
    const uint32_t* __restrict__ d_cagra_labels,
    const int*          __restrict__ d_inv_list,
    const gpu_cluster_pos_t* __restrict__ d_cluster_pos,
    const int*          __restrict__ d_query_offsets,
    size_t*             d_emb_ids,
    int                 nprobe,
    size_t              n_clusters,
    size_t              num_queries,
    bool                use_clustered_layout
);

__global__ void gather_doc_lengths_kernel(
    const int* __restrict__ d_topk_doc_ids,
    const int* __restrict__ d_doc_ptrs,
    int*       d_doc_lengths,
    size_t k
);

__global__ void aggregate_stage1_tracked_kernel(
    const size_t* __restrict__ d_emb_ids,
    const float*  __restrict__ d_emb_dists,
    const int*    __restrict__ d_pair_offsets,
    const int*    __restrict__ d_doc_ids,
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
);

__global__ void sum_doc_scores_sparse_kernel(
    const float* __restrict__ d_doc_query_max,
    const int*   __restrict__ d_touched_doc_list,
    float*       d_scores_out,
    int*         d_doc_ids_out,
    int          num_touched
#ifndef GPU_MVR_COMPACT_DOC_BUFFER
    , size_t     num_docs
#endif
);
