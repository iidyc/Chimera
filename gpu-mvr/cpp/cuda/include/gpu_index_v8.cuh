#pragma once

#define gpu_mvr_index gpu_mvr_index_v8_base
#define cast_int_size_t cast_int_size_t_v8_base
#include "gpu_index_v3.cuh"
#undef cast_int_size_t
#undef gpu_mvr_index

#include "gpu_kernels_v8.cuh"

struct gpu_mvr_index : public gpu_mvr_index_v8_base {
    gpu_mvr_index(
        const std::string& filename,
        const std::vector<int>& doc_lens,
        const gpu_search_runtime_options& runtime_options);
    ~gpu_mvr_index() = default;

    std::vector<size_t> search(const float* queries, size_t k);
    std::vector<size_t> search_profiled(const float* queries, size_t k);
    template <bool kProfile>
    std::vector<size_t> search_impl(const float* queries, size_t k);

    void rank_cluster_dists_gpu(
        query_object* h_query_objs,
        size_t nprobe,
        size_t k,
        int& actual_k_out,
        cudaStream_t stream = 0);
    template <bool kProfile>
    void rank_cluster_dists_gpu_impl(
        query_object* h_query_objs,
        size_t nprobe,
        size_t k,
        int& actual_k_out,
        cudaStream_t stream = 0);
};
