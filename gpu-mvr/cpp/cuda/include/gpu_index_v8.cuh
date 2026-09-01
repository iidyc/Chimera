#pragma once

#include <memory>

#define gpu_mvr_index gpu_mvr_index_v8_base
#define cast_int_size_t cast_int_size_t_v8_base
#include "gpu_index_v3.cuh"
#undef cast_int_size_t
#undef gpu_mvr_index

#include "gpu_kernels_v8.cuh"

struct gpu_search_profile_v8 {
    double search_wall_ms = 0.0;
    double total_search_time_ms = 0.0;
    double stage1_time_ms = 0.0;
    double stage2_time_ms = 0.0;
    double s1_cagra_ms = 0.0;
    double s1_expansion_ms = 0.0;
    double s1_binary_ip_ms = 0.0;
    double s1_atomic_agg_ms = 0.0;
    double s1_sum_scores_ms = 0.0;
    double s1_topk_sort_ms = 0.0;
    double s1_d2d_ms = 0.0;
    double s1_memset_ms = 0.0;
    double s1_sum_accounted_ms = 0.0;
    double phase_a_wall_ms = 0.0;
    double phase_a_gpu_total_ms = 0.0;
    double phase_a_gather_ms = 0.0;
    double phase_a_prefix_ms = 0.0;
    double phase_a_token_ids_ms = 0.0;
    double phase_a_d2h_ms = 0.0;
    double phase_a_cpu_launch_gather_ms = 0.0;
    double phase_a_cpu_launch_prefix_ms = 0.0;
    double phase_a_cpu_launch_token_ids_ms = 0.0;
    double phase_a_cpu_launch_d2h_ms = 0.0;
    double phase_a_cpu_sync_ms = 0.0;
    double phase_a_cpu_event_elapsed_ms = 0.0;
    double phase_a_gpu_sum_accounted_ms = 0.0;
    double phase_b_wall_ms = 0.0;
    double phase_b_binary_ip_total_ms = 0.0;
    double phase_b_doc_score_total_ms = 0.0;
    double phase_b_total_kernel_ms = 0.0;
    double phase_c_wait_d2h_ms = 0.0;
    double phase_c_topk_ms = 0.0;
    double phase_c_identify_ms = 0.0;
    double phase_c_prepare_ms = 0.0;
    double phase_c_cpu_refine_ms = 0.0;
    double phase_c_combine_ms = 0.0;
    double phase_c_total_ms = 0.0;
    double phase_c_refined_docs = 0.0;
    double phase_abc_total_wall_ms = 0.0;
    double transfer_h2d_ms = 0.0;
    double transfer_d2h_ms = 0.0;
    double transfer_total_ms = 0.0;
    double transfer_h2d_bytes = 0.0;
    double transfer_d2h_bytes = 0.0;
    double transfer_count = 0.0;
};

void accumulate_gpu_search_profile_v8(gpu_search_profile_v8& total, const gpu_search_profile_v8& sample);
void average_gpu_search_profile_v8(gpu_search_profile_v8& profile, double count);
void print_gpu_search_profile_v8(const gpu_search_profile_v8& profile, const char* prefix = "[PROFILE]");

struct gpu_mvr_index : public gpu_mvr_index_v8_base {
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

    gpu_mvr_index(
        const std::string& filename,
        const std::vector<int>& doc_lens,
        const gpu_search_runtime_options& runtime_options);
    gpu_mvr_index(
        gpu_mvr_index& owner,
        const gpu_search_runtime_options& runtime_options);
    ~gpu_mvr_index();

    bool is_query_slot_ = false;
#ifdef GPU_MVR_HAVE_CUVS
    raft::resources cagra_res_;
#endif

    void compute_workspace_probe_bounds();
    void allocate_workspace();
    void ensure_compact_doc_capacity(size_t required_rows);

    std::vector<size_t> search(const float* queries, size_t k);
    std::vector<size_t> search_profiled(const float* queries, size_t k);
    std::vector<size_t> search_profiled(
        const float* queries,
        size_t k,
        gpu_search_profile_v8* profile,
        bool print_profile = true);
    template <bool kProfile>
    std::vector<size_t> search_impl(
        const float* queries,
        size_t k,
        gpu_search_profile_v8* profile = nullptr,
        bool print_profile = true);

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
