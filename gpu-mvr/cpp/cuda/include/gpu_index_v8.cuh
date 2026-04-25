#pragma once

#define gpu_mvr_index gpu_mvr_index_v8_base
#define cast_int_size_t cast_int_size_t_v8_base
#include "gpu_index_v3.cuh"
#undef cast_int_size_t
#undef gpu_mvr_index

#include "gpu_kernels_v8.cuh"

struct gpu_search_profile_v8 {
    double total_search_time_ms = 0.0;
    double search_wall_ms = 0.0;

    double stage1_time_ms = 0.0;
    double s1_cagra_ms = 0.0;
    double s1_expansion_ms = 0.0;
    double s1_binary_ip_ms = 0.0;
    double s1_atomic_agg_ms = 0.0;
    double s1_sum_scores_ms = 0.0;
    double s1_topk_sort_ms = 0.0;
    double s1_d2d_ms = 0.0;
    double s1_memset_ms = 0.0;
    double s1_sum_accounted_ms = 0.0;

    double stage23_time_ms = 0.0;
    double phase_b_binary_ip_total_ms = 0.0;
    double phase_b_doc_score_total_ms = 0.0;
    double phase_b_total_kernel_ms = 0.0;

    double total_h2d_ms = 0.0;
    double total_d2h_ms = 0.0;
    double total_transfer_ms = 0.0;
    double total_h2d_bytes = 0.0;
    double total_d2h_bytes = 0.0;
    double total_transfer_bytes = 0.0;
};

inline void accumulate_gpu_search_profile_v8(
    gpu_search_profile_v8& dst,
    const gpu_search_profile_v8& src)
{
#define GPU_MVR_V8_ACCUM_FIELD(name) dst.name += src.name
    GPU_MVR_V8_ACCUM_FIELD(total_search_time_ms);
    GPU_MVR_V8_ACCUM_FIELD(search_wall_ms);
    GPU_MVR_V8_ACCUM_FIELD(stage1_time_ms);
    GPU_MVR_V8_ACCUM_FIELD(s1_cagra_ms);
    GPU_MVR_V8_ACCUM_FIELD(s1_expansion_ms);
    GPU_MVR_V8_ACCUM_FIELD(s1_binary_ip_ms);
    GPU_MVR_V8_ACCUM_FIELD(s1_atomic_agg_ms);
    GPU_MVR_V8_ACCUM_FIELD(s1_sum_scores_ms);
    GPU_MVR_V8_ACCUM_FIELD(s1_topk_sort_ms);
    GPU_MVR_V8_ACCUM_FIELD(s1_d2d_ms);
    GPU_MVR_V8_ACCUM_FIELD(s1_memset_ms);
    GPU_MVR_V8_ACCUM_FIELD(s1_sum_accounted_ms);
    GPU_MVR_V8_ACCUM_FIELD(stage23_time_ms);
    GPU_MVR_V8_ACCUM_FIELD(phase_b_binary_ip_total_ms);
    GPU_MVR_V8_ACCUM_FIELD(phase_b_doc_score_total_ms);
    GPU_MVR_V8_ACCUM_FIELD(phase_b_total_kernel_ms);
    GPU_MVR_V8_ACCUM_FIELD(total_h2d_ms);
    GPU_MVR_V8_ACCUM_FIELD(total_d2h_ms);
    GPU_MVR_V8_ACCUM_FIELD(total_transfer_ms);
    GPU_MVR_V8_ACCUM_FIELD(total_h2d_bytes);
    GPU_MVR_V8_ACCUM_FIELD(total_d2h_bytes);
    GPU_MVR_V8_ACCUM_FIELD(total_transfer_bytes);
#undef GPU_MVR_V8_ACCUM_FIELD
}

inline void average_gpu_search_profile_v8(
    gpu_search_profile_v8& profile,
    double count)
{
    if (count <= 0.0) {
        return;
    }
#define GPU_MVR_V8_AVG_FIELD(name) profile.name /= count
    GPU_MVR_V8_AVG_FIELD(total_search_time_ms);
    GPU_MVR_V8_AVG_FIELD(search_wall_ms);
    GPU_MVR_V8_AVG_FIELD(stage1_time_ms);
    GPU_MVR_V8_AVG_FIELD(s1_cagra_ms);
    GPU_MVR_V8_AVG_FIELD(s1_expansion_ms);
    GPU_MVR_V8_AVG_FIELD(s1_binary_ip_ms);
    GPU_MVR_V8_AVG_FIELD(s1_atomic_agg_ms);
    GPU_MVR_V8_AVG_FIELD(s1_sum_scores_ms);
    GPU_MVR_V8_AVG_FIELD(s1_topk_sort_ms);
    GPU_MVR_V8_AVG_FIELD(s1_d2d_ms);
    GPU_MVR_V8_AVG_FIELD(s1_memset_ms);
    GPU_MVR_V8_AVG_FIELD(s1_sum_accounted_ms);
    GPU_MVR_V8_AVG_FIELD(stage23_time_ms);
    GPU_MVR_V8_AVG_FIELD(phase_b_binary_ip_total_ms);
    GPU_MVR_V8_AVG_FIELD(phase_b_doc_score_total_ms);
    GPU_MVR_V8_AVG_FIELD(phase_b_total_kernel_ms);
    GPU_MVR_V8_AVG_FIELD(total_h2d_ms);
    GPU_MVR_V8_AVG_FIELD(total_d2h_ms);
    GPU_MVR_V8_AVG_FIELD(total_transfer_ms);
    GPU_MVR_V8_AVG_FIELD(total_h2d_bytes);
    GPU_MVR_V8_AVG_FIELD(total_d2h_bytes);
    GPU_MVR_V8_AVG_FIELD(total_transfer_bytes);
#undef GPU_MVR_V8_AVG_FIELD
}

inline void print_gpu_search_profile_v8(
    const gpu_search_profile_v8& profile,
    const char* prefix = "[PROFILE]")
{
    std::cout << prefix << " Mode: Persistent Stage 2+3 (v8 clustered LUT path)\n";
    std::cout << prefix << " Stage 1 time: " << profile.stage1_time_ms << " ms\n";
    std::cout << prefix << "   1. CAGRA search            : " << profile.s1_cagra_ms << " ms\n";
    std::cout << prefix << "   2. GPU IVF expansion       : " << profile.s1_expansion_ms << " ms\n";
    std::cout << prefix << "   3. Binary IP kernel        : " << profile.s1_binary_ip_ms << " ms\n";
    std::cout << prefix << "   4. Aggregation + tracking  : " << profile.s1_atomic_agg_ms << " ms\n";
    std::cout << prefix << "   5. Sum doc scores (sparse) : " << profile.s1_sum_scores_ms << " ms\n";
    std::cout << prefix << "   6. Top-k sort (sparse)     : " << profile.s1_topk_sort_ms << " ms\n";
    std::cout << prefix << "   7. D2D copy top-k doc IDs  : " << profile.s1_d2d_ms << " ms\n";
    std::cout << prefix << "   8. Memset (overlapped 1-3) : " << profile.s1_memset_ms
              << " ms (not in critical path)\n";
    std::cout << prefix << "   Sum accounted              : " << profile.s1_sum_accounted_ms << " ms\n";
    std::cout << prefix << " Stage 2+3 time              : " << profile.stage23_time_ms << " ms\n";
    std::cout << prefix << " Phase B binary_ip total     : " << profile.phase_b_binary_ip_total_ms << " ms\n";
    std::cout << prefix << " Phase B doc_score total     : " << profile.phase_b_doc_score_total_ms << " ms\n";
    std::cout << prefix << " Phase B total kernel time   : " << profile.phase_b_total_kernel_ms << " ms\n";
    std::cout << prefix << " Total search time           : " << profile.total_search_time_ms << " ms\n";
    std::cout << prefix << " Data transfer summary:\n";
    std::cout << prefix << "   H2D: " << profile.total_h2d_ms << " ms, "
              << static_cast<unsigned long long>(std::llround(profile.total_h2d_bytes))
              << " bytes (" << (profile.total_h2d_bytes / 1024.0) << " KB)\n";
    std::cout << prefix << "   D2H: " << profile.total_d2h_ms << " ms, "
              << static_cast<unsigned long long>(std::llround(profile.total_d2h_bytes))
              << " bytes (" << (profile.total_d2h_bytes / 1024.0) << " KB)\n";
    std::cout << prefix << "   Total transfer: " << profile.total_transfer_ms << " ms, "
              << static_cast<unsigned long long>(std::llround(profile.total_transfer_bytes))
              << " bytes (" << (profile.total_transfer_bytes / 1024.0) << " KB)\n";
    std::cout << prefix << " Total wall-clock time: " << profile.search_wall_ms << " ms\n";
}

struct gpu_mvr_index : public gpu_mvr_index_v8_base {
    gpu_mvr_index(
        const std::string& filename,
        const std::vector<int>& doc_lens,
        const gpu_search_runtime_options& runtime_options);
    ~gpu_mvr_index() = default;

    std::vector<size_t> search(const float* queries, size_t k);
    std::vector<size_t> search_profiled(const float* queries, size_t k);
    std::vector<size_t> search_profiled(
        const float* queries,
        size_t k,
        gpu_search_profile_v8* profile,
        bool print_profile);
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
