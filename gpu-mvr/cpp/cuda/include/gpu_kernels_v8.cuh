#pragma once

#include <cuda_runtime.h>

#include "gpu_kernels_v3.cuh"

inline constexpr size_t kStage1ClusterAlignTokensV8 = 8;
inline constexpr int kStage1MaxNprobeV8 = 512;

__global__ void stage1_binary_ip_lut_v8_align_kernel(
    const float*    __restrict__ d_lut,
    const char*     __restrict__ d_clustered_code,
    const float*    __restrict__ d_clustered_factor,
    const float*    __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    float*          d_doc_query_max,
    int*            d_doc_touched,
    int*            d_touched_doc_list,
    int*            d_num_touched_docs,
    size_t          num_docs,
    int             nprobe,
    size_t          n_clusters
);
