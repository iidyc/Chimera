#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "gpu_config.cuh"

inline constexpr int STAGE2_LUT_TILE_Q = 8;
inline constexpr int STAGE2_LUT_NUM_TILES = Q_DOCLEN / STAGE2_LUT_TILE_Q;
inline constexpr int STAGE2_LUT_SMEM_FLOATS = STAGE2_LUT_TILE_Q * LUT_ENTRIES_PER_QUERY;

__global__ void stage1_binary_ip_kernel_v2(
    const float* __restrict__ d_queries,
    const char* __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_emb_ids,
    const int* __restrict__ d_pair_offsets,
    float* __restrict__ d_out_dists,
    size_t max_embs_per_query);

__global__ void stage2_binary_ip_kernel_v2(
    const float* __restrict__ d_queries,
    const char* __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_token_ids,
    float* __restrict__ d_out_dists,
    size_t total_tokens,
    size_t batch_tokens);

__global__ void precompute_lut_kernel(
    const float* __restrict__ d_queries,
    float* __restrict__ d_lut);

__global__ void stage1_binary_ip_lut_kernel(
    const float* __restrict__ d_lut,
    const char* __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_emb_ids,
    const int* __restrict__ d_pair_offsets,
    float* __restrict__ d_out_dists,
    size_t max_embs_per_query);

__global__ void stage2_binary_ip_lut_kernel(
    const float* __restrict__ d_lut,
    const char* __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_token_ids,
    float* __restrict__ d_out_dists,
    size_t total_tokens,
    size_t batch_tokens);

namespace gpu_mvr::v1 {
__global__ void doc_score(
    const float* __restrict__ d_token_dists,
    const size_t* __restrict__ d_candidate_offsets,
    float* d_doc_scores,
    size_t total_tokens,
    size_t num_candidates);

void launch_doc_score(
    const float* d_token_dists,
    const size_t* d_candidate_offsets,
    float* d_doc_scores,
    float* d_doc_query_max,
    size_t total_tokens,
    size_t num_candidates,
    int score_threads,
    cudaStream_t stream);
}  // namespace gpu_mvr::v1

namespace gpu_mvr::v2 {
__global__ void doc_score_max(
    const float* __restrict__ d_token_dists,
    const size_t* __restrict__ d_candidate_offsets,
    float* d_doc_query_max,
    size_t total_tokens,
    size_t num_candidates);

__global__ void doc_score(
    const float* __restrict__ d_doc_query_max,
    float* d_doc_scores,
    size_t num_candidates);

void launch_doc_score(
    const float* d_token_dists,
    const size_t* d_candidate_offsets,
    float* d_doc_scores,
    float* d_doc_query_max,
    size_t total_tokens,
    size_t num_candidates,
    int score_threads,
    cudaStream_t stream);
}  // namespace gpu_mvr::v2

namespace gpu_mvr {
void configure_binary_ip_kernels();

void precompute_binary_ip_state(
    const float* d_queries,
    float* d_lut,
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

void launch_doc_score(
    const float* d_token_dists,
    const size_t* d_candidate_offsets,
    float* d_doc_scores,
    float* d_doc_query_max,
    size_t total_tokens,
    size_t num_candidates,
    int score_threads,
    cudaStream_t stream);
}  // namespace gpu_mvr

__global__ void extract_one_bit_dists_kernel(
    const float* __restrict__ d_token_dists,
    const size_t* __restrict__ d_candidate_offsets,
    const int* __restrict__ d_selected_indices,
    float* d_out_one_bit_dists,
    const size_t* __restrict__ d_out_offsets,
    size_t total_tokens,
    size_t k);

__global__ void gather_token_ids_kernel(
    const int* __restrict__ d_candidate_doc_ids,
    const int* __restrict__ d_doc_ptrs,
    const size_t* __restrict__ d_candidate_offsets,
    size_t* d_out_token_ids,
    size_t num_candidates);

__global__ void compute_query_expansion_sizes_kernel(
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t* __restrict__ d_cluster_pos,
    int* d_query_sizes,
    int nprobe,
    size_t n_clusters,
    size_t num_queries);

__global__ void expand_cluster_ids_kernel(
    const uint32_t* __restrict__ d_cagra_labels,
    const int* __restrict__ d_inv_list,
    const size_t* __restrict__ d_cluster_pos,
    const int* __restrict__ d_query_offsets,
    size_t* d_emb_ids,
    int nprobe,
    size_t n_clusters,
    size_t num_queries);

__global__ void gather_doc_lengths_kernel(
    const int* __restrict__ d_topk_doc_ids,
    const int* __restrict__ d_doc_ptrs,
    int* d_doc_lengths,
    size_t k);

__device__ __forceinline__ float atomicMaxFloat(float* address, float val);

__global__ void aggregate_stage1_tracked_kernel(
    const size_t* __restrict__ d_emb_ids,
    const float* __restrict__ d_emb_dists,
    const int* __restrict__ d_pair_offsets,
    const int* __restrict__ d_doc_ids,
    float* d_doc_query_max,
    int* d_doc_touched,
    int* d_touched_doc_list,
    int* d_num_touched_docs,
    size_t num_docs,
    size_t total_pairs,
    size_t max_embs_per_query);

__global__ void sum_doc_scores_kernel(
    const float* __restrict__ d_doc_query_max,
    float* d_doc_scores,
    size_t num_docs);

__global__ void sum_doc_scores_sparse_kernel(
    const float* __restrict__ d_doc_query_max,
    const int* __restrict__ d_touched_doc_list,
    float* d_scores_out,
    int* d_doc_ids_out,
    int num_touched,
    size_t num_docs);

__global__ void cleanup_touched_docs_kernel(
    float* d_doc_query_max,
    int* d_doc_touched,
    const int* __restrict__ d_touched_doc_list,
    int num_touched,
    size_t num_docs);
