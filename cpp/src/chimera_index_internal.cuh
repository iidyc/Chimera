#pragma once

#include <cuda_runtime.h>

#include "chimera/chimera_index.cuh"

#include <numeric>
#include <utility>
#include <vector>

namespace Chimera {

struct chimera_index::Workspace {
    float* d_queries = nullptr;
    float* d_cb1_sumq = nullptr;
    int* d_refined_doc_lengths = nullptr;
    float* d_one_bit_token_scores = nullptr;
    float* d_one_bit_document_scores = nullptr;
    float* d_lut = nullptr;
    float* d_cagra_dists = nullptr;
    uint32_t* d_cagra_labels = nullptr;
    int* d_sorted_candidate_doc_ids = nullptr;
    int* d_materialized_candidate_doc_ids = nullptr;
    float* d_partial_scores = nullptr;
    float* d_partial_query_scores = nullptr;
    void* d_cub_temp_storage = nullptr;
    size_t cub_temp_storage_bytes = 0;
    float* d_sorted_partial_scores = nullptr;
    int* d_candidate_doc_ids = nullptr;
    float* h_pinned_queries = nullptr;
    float* h_pinned_cb1_sumq = nullptr;
    size_t max_refined_candidates = 0;
    size_t max_refined_tokens = 0;
    size_t max_full_bit_candidates = 0;
    size_t* d_refined_doc_offsets = nullptr;
    uint32_t* d_refined_token_positions = nullptr;
    float* h_mapped_one_bit_document_scores = nullptr;
    size_t* h_pinned_refined_doc_offsets = nullptr;
    int* h_refined_doc_ids = nullptr;
    size_t* h_num_refined_tokens = nullptr;
    cudaStream_t stream_compute = nullptr;
    cudaStream_t stream_h2d = nullptr;
    cudaStream_t stream_d2h = nullptr;
    cudaEvent_t event_h2d_done = nullptr;
    std::vector<int> one_bit_score_order;
    std::vector<int> full_bit_candidate_indices;
    std::vector<std::pair<float, int>> full_bit_scores;
    std::vector<bool> seen_doc_map;
};

struct chimera_index::QueryState {
    float* rotated_query;
    float cb1_sumq;
    float cbex_sumq;

    QueryState() = default;

    explicit QueryState(float* rotated_query, size_t padded_dim, size_t ex_bits)
        : rotated_query(rotated_query) {
        const float sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, 0.0f);
        cb1_sumq = sumq * ((1 << 1) - 1) / 2.0f;
        cbex_sumq = sumq * ((1 << (ex_bits + 1)) - 1) / 2.0f;
    }
};

}  // namespace Chimera
