#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "gpu_kernels_v3.cuh"

namespace Chimera {

using doc_bitmap_bucket_t = uint32_t;
using doc_bitmap_offset_t = uint32_t;

inline constexpr int kDocBitmapBucketWidth = 32;
inline constexpr int kDocBitmapCompRatio = 8;
inline constexpr size_t kStage1ClusterAlignTokens = 8;
inline constexpr int kStage1MaxNprobe = 512;

inline constexpr size_t doc_bitmap_num_buckets(size_t num_docs) {
    const size_t raw =
        (num_docs + kDocBitmapBucketWidth - 1) / kDocBitmapBucketWidth;
    return ((raw + kDocBitmapCompRatio - 1) / kDocBitmapCompRatio) *
           kDocBitmapCompRatio;
}

inline constexpr size_t doc_bitmap_num_offsets(size_t num_buckets) {
    return (num_buckets / kDocBitmapCompRatio) + 1;
}

__global__ void stage1_binary_ip_lut_v7_lite_kernel(
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
);

__global__ void stage1_binary_ip_lut_flag_docs_v7_lite_kernel(
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    size_t          num_docs,
    int             nprobe,
    size_t          n_clusters
);

__global__ void bitmap_offset_init_v7_lite_kernel(
    const doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    size_t num_buckets,
    doc_bitmap_offset_t* __restrict__ d_doc_bitmap_offsets
);

__global__ void bitmap_unique_docs_v7_lite_kernel(
    const doc_bitmap_bucket_t* __restrict__ d_doc_bitmap,
    size_t num_buckets,
    const doc_bitmap_offset_t* __restrict__ d_doc_bitmap_offsets,
    int* __restrict__ d_out_doc_ids
);

__global__ void sum_doc_scores_compact_v7_lite_kernel(
    const float* __restrict__ d_doc_query_max,
    const int*   __restrict__ d_touched_doc_list,
    float*       d_scores_out,
    int*         d_doc_ids_out,
    int          num_touched
);


}  // namespace Chimera
