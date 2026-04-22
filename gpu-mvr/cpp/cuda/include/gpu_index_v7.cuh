#pragma once

#define gpu_mvr_index gpu_mvr_index_v7_base
#define cast_int_size_t cast_int_size_t_v7_base
#include "gpu_index_v3.cuh"
#undef cast_int_size_t
#undef gpu_mvr_index

#include "gpu_kernels_v7.cuh"

struct gpu_mvr_index : public gpu_mvr_index_v7_base {
    enum class Stage1AlignMode {
        CompactAlign,
        MemoryPaddedAlign,
    };

    Stage1AlignMode stage1_align_mode_ = Stage1AlignMode::CompactAlign;
    size_t workspace_probe_unique_doc_bound_ = 0;

    doc_bitmap_bucket_t* d_doc_bitmap_ = nullptr;
    doc_bitmap_offset_t* d_doc_bitmap_offsets_ = nullptr;
    size_t doc_bitmap_bucket_count_ = 0;
    size_t doc_bitmap_offset_count_ = 0;
    size_t max_compact_docs_ = 0;
    size_t compact_doc_capacity_ = 0;
    size_t compact_topk_capacity_ = 0;
    char* d_compact_doc_buffer_raw_ = nullptr;
    char* d_compact_doc_buffer_ = nullptr;
    size_t compact_doc_buffer_bytes_ = 0;

    char* d_stage1_padded_code_ = nullptr;
    float* d_stage1_padded_factor_ = nullptr;
    int* d_stage1_padded_doc_ids_ = nullptr;
    uint2* d_stage1_padded_cluster_meta_ = nullptr;
    size_t stage1_padded_tokens_ = 0;

    gpu_mvr_index(
        const std::string& filename,
        const std::vector<int>& doc_lens,
        const gpu_search_runtime_options& runtime_options);
    ~gpu_mvr_index();

    void compute_workspace_probe_bounds();
    void allocate_workspace();
    void ensure_compact_doc_capacity(size_t required_rows);
    void maybe_load_stage1_padded_layout(const std::string& filename);

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
