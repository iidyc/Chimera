#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "chimera/config.cuh"

// Query-tiled one-bit scoring constants used during document scoring.
#define DOCUMENT_SCORING_LUT_TILE_Q 8
#define DOCUMENT_SCORING_LUT_NUM_TILES \
    (Q_DOCLEN / DOCUMENT_SCORING_LUT_TILE_Q)
#define DOCUMENT_SCORING_LUT_SMEM_FLOATS \
    (DOCUMENT_SCORING_LUT_TILE_Q * LUT_ENTRIES_PER_QUERY)

namespace Chimera {

__global__ void precompute_one_bit_lut_kernel(
    const float* __restrict__ d_queries,
    float* __restrict__ d_lut
);

__global__ void one_bit_token_score_kernel(
    const float* __restrict__ d_lut,
    const char*  __restrict__ d_clustered_code,
    const float* __restrict__ d_clustered_factor,
    const float* __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_clustered_pos,
    float* __restrict__ d_one_bit_scores,
    size_t total_tokens,
    size_t batch_tokens
);

__global__ void one_bit_document_score_kernel(
    const float*  __restrict__ d_one_bit_token_scores,
    const size_t* __restrict__ d_refined_doc_offsets,
    float*        d_one_bit_document_scores,
    size_t total_tokens,
    size_t num_refined_candidates
);

__global__ void gather_refined_token_positions_kernel(
    const int*    __restrict__ d_refined_doc_ids,
    const int*    __restrict__ d_doc_ptrs,
    const uint32_t* __restrict__ d_token_to_cluster_pos,
    const size_t* __restrict__ d_refined_doc_offsets,
    uint32_t*     d_refined_token_positions,
    size_t num_refined_candidates
);

__global__ void gather_refined_doc_lengths_kernel(
    const int* __restrict__ d_refined_doc_ids,
    const int* __restrict__ d_doc_ptrs,
    int*       d_doc_lengths,
    size_t num_refined_candidates
);

using candidate_bitmap_bucket_t = uint32_t;
using candidate_bitmap_offset_t = uint32_t;

inline constexpr int kCandidateBitmapBucketWidth = 32;
inline constexpr int kCandidateBitmapCompressionRatio = 8;
inline constexpr size_t kCandidateRefinementClusterAlignment = 8;
inline constexpr int kCandidateRefinementMaxNprobe = 512;

inline constexpr size_t candidate_bitmap_num_buckets(size_t num_docs) {
    const size_t raw =
        (num_docs + kCandidateBitmapBucketWidth - 1) /
        kCandidateBitmapBucketWidth;
    return ((raw + kCandidateBitmapCompressionRatio - 1) /
            kCandidateBitmapCompressionRatio) *
           kCandidateBitmapCompressionRatio;
}

inline constexpr size_t candidate_bitmap_num_offsets(size_t num_buckets) {
    return (num_buckets / kCandidateBitmapCompressionRatio) + 1;
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
);


__global__ void mark_candidate_documents_kernel(
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    size_t          num_docs,
    int             nprobe,
    size_t          n_clusters
);

__global__ void count_candidate_bitmap_kernel(
    const candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    size_t num_buckets,
    candidate_bitmap_offset_t* __restrict__ d_candidate_bitmap_offsets
);

__global__ void materialize_candidate_documents_kernel(
    const candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    size_t num_buckets,
    const candidate_bitmap_offset_t* __restrict__ d_candidate_bitmap_offsets,
    int* __restrict__ d_candidate_doc_ids
);

__global__ void sum_partial_scores_kernel(
    const float* __restrict__ d_partial_query_scores,
    const int*   __restrict__ d_candidate_doc_ids,
    float*       d_partial_scores,
    int*         d_scored_doc_ids,
    int          num_candidate_docs
);


}  // namespace Chimera
