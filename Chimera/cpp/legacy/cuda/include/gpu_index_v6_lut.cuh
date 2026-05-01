#pragma once

#ifndef CHIMERA_USE_LUT
#define CHIMERA_USE_LUT 1
#endif

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/adjacent_difference.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cub/cub.cuh>

// Prevent Eigen from adding __host__ __device__ annotations to its functions.
#define EIGEN_NO_CUDA

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cfloat>
#include <limits>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <immintrin.h>
#include "gpu_search_options.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "quantization.hpp"
#include "estimator.hpp"
#include "query.hpp"
#include "ivf_pg.hpp"
#include "gpu_config.cuh"
#include "gpu_kernels_v6_lut.cuh"

namespace Chimera {

using namespace rabitqlib;

#ifndef CHIMERA_CUDA_CHECK_DEFINED
#define CHIMERA_CUDA_CHECK_DEFINED
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// === Transfer profiling macros ===
#ifdef CHIMERA_PROFILE
#define XFER_RECORD_BEGIN(stream) \
    do { if (ws_.xfer_count < Workspace::MAX_XFER_RECORDS) \
        CUDA_CHECK(cudaEventRecord(ws_.xfer_records[ws_.xfer_count].start, (stream))); } while(0)

#define XFER_RECORD_END(stream, nbytes, h2d_flag) \
    do { if (ws_.xfer_count < Workspace::MAX_XFER_RECORDS) { \
        CUDA_CHECK(cudaEventRecord(ws_.xfer_records[ws_.xfer_count].end, (stream))); \
        ws_.xfer_records[ws_.xfer_count].bytes = (size_t)(nbytes); \
        ws_.xfer_records[ws_.xfer_count].is_h2d = (h2d_flag); \
        ws_.xfer_count++; } } while(0)
#else
#define XFER_RECORD_BEGIN(stream)
#define XFER_RECORD_END(stream, nbytes, h2d_flag)
#endif

struct cast_int_size_t {
    __host__ __device__ size_t operator()(int x) const { return (size_t)x; };
};

struct gpu_search_profile_v6 {
    bool persistent_stage23 = false;

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
    double phase_c_gpu_extract_ms = 0.0;
    double phase_c_cpu_ip_ex_ms = 0.0;
    double phase_c_wait_extract_ms = 0.0;
    double phase_c_combine_ms = 0.0;
    double phase_c_total_ms = 0.0;
    double phase_c_refined_docs = 0.0;
    double phase_abc_total_wall_ms = 0.0;

    double stage2_time_ms = 0.0;
    double stage3_time_ms = 0.0;
    double stage3_docs = 0.0;

    double total_h2d_ms = 0.0;
    double total_d2h_ms = 0.0;
    double total_transfer_ms = 0.0;
    double total_h2d_bytes = 0.0;
    double total_d2h_bytes = 0.0;
    double total_transfer_bytes = 0.0;
};

inline void accumulate_gpu_search_profile_v6(
    gpu_search_profile_v6& dst,
    const gpu_search_profile_v6& src)
{
    dst.persistent_stage23 = src.persistent_stage23;
#define CHIMERA_ACCUM_FIELD(name) dst.name += src.name
    CHIMERA_ACCUM_FIELD(total_search_time_ms);
    CHIMERA_ACCUM_FIELD(search_wall_ms);
    CHIMERA_ACCUM_FIELD(stage1_time_ms);
    CHIMERA_ACCUM_FIELD(s1_cagra_ms);
    CHIMERA_ACCUM_FIELD(s1_expansion_ms);
    CHIMERA_ACCUM_FIELD(s1_binary_ip_ms);
    CHIMERA_ACCUM_FIELD(s1_atomic_agg_ms);
    CHIMERA_ACCUM_FIELD(s1_sum_scores_ms);
    CHIMERA_ACCUM_FIELD(s1_topk_sort_ms);
    CHIMERA_ACCUM_FIELD(s1_d2d_ms);
    CHIMERA_ACCUM_FIELD(s1_memset_ms);
    CHIMERA_ACCUM_FIELD(s1_sum_accounted_ms);
    CHIMERA_ACCUM_FIELD(phase_a_wall_ms);
    CHIMERA_ACCUM_FIELD(phase_a_gpu_total_ms);
    CHIMERA_ACCUM_FIELD(phase_a_gather_ms);
    CHIMERA_ACCUM_FIELD(phase_a_prefix_ms);
    CHIMERA_ACCUM_FIELD(phase_a_token_ids_ms);
    CHIMERA_ACCUM_FIELD(phase_a_d2h_ms);
    CHIMERA_ACCUM_FIELD(phase_a_cpu_launch_gather_ms);
    CHIMERA_ACCUM_FIELD(phase_a_cpu_launch_prefix_ms);
    CHIMERA_ACCUM_FIELD(phase_a_cpu_launch_token_ids_ms);
    CHIMERA_ACCUM_FIELD(phase_a_cpu_launch_d2h_ms);
    CHIMERA_ACCUM_FIELD(phase_a_cpu_sync_ms);
    CHIMERA_ACCUM_FIELD(phase_a_cpu_event_elapsed_ms);
    CHIMERA_ACCUM_FIELD(phase_a_gpu_sum_accounted_ms);
    CHIMERA_ACCUM_FIELD(phase_b_wall_ms);
    CHIMERA_ACCUM_FIELD(phase_b_binary_ip_total_ms);
    CHIMERA_ACCUM_FIELD(phase_b_doc_score_total_ms);
    CHIMERA_ACCUM_FIELD(phase_b_total_kernel_ms);
    CHIMERA_ACCUM_FIELD(phase_c_wait_d2h_ms);
    CHIMERA_ACCUM_FIELD(phase_c_topk_ms);
    CHIMERA_ACCUM_FIELD(phase_c_identify_ms);
    CHIMERA_ACCUM_FIELD(phase_c_gpu_extract_ms);
    CHIMERA_ACCUM_FIELD(phase_c_cpu_ip_ex_ms);
    CHIMERA_ACCUM_FIELD(phase_c_wait_extract_ms);
    CHIMERA_ACCUM_FIELD(phase_c_combine_ms);
    CHIMERA_ACCUM_FIELD(phase_c_total_ms);
    CHIMERA_ACCUM_FIELD(phase_c_refined_docs);
    CHIMERA_ACCUM_FIELD(phase_abc_total_wall_ms);
    CHIMERA_ACCUM_FIELD(stage2_time_ms);
    CHIMERA_ACCUM_FIELD(stage3_time_ms);
    CHIMERA_ACCUM_FIELD(stage3_docs);
    CHIMERA_ACCUM_FIELD(total_h2d_ms);
    CHIMERA_ACCUM_FIELD(total_d2h_ms);
    CHIMERA_ACCUM_FIELD(total_transfer_ms);
    CHIMERA_ACCUM_FIELD(total_h2d_bytes);
    CHIMERA_ACCUM_FIELD(total_d2h_bytes);
    CHIMERA_ACCUM_FIELD(total_transfer_bytes);
#undef CHIMERA_ACCUM_FIELD
}

inline void average_gpu_search_profile_v6(
    gpu_search_profile_v6& profile,
    double count)
{
    if (count <= 0.0) {
        return;
    }
#define CHIMERA_AVG_FIELD(name) profile.name /= count
    CHIMERA_AVG_FIELD(total_search_time_ms);
    CHIMERA_AVG_FIELD(search_wall_ms);
    CHIMERA_AVG_FIELD(stage1_time_ms);
    CHIMERA_AVG_FIELD(s1_cagra_ms);
    CHIMERA_AVG_FIELD(s1_expansion_ms);
    CHIMERA_AVG_FIELD(s1_binary_ip_ms);
    CHIMERA_AVG_FIELD(s1_atomic_agg_ms);
    CHIMERA_AVG_FIELD(s1_sum_scores_ms);
    CHIMERA_AVG_FIELD(s1_topk_sort_ms);
    CHIMERA_AVG_FIELD(s1_d2d_ms);
    CHIMERA_AVG_FIELD(s1_memset_ms);
    CHIMERA_AVG_FIELD(s1_sum_accounted_ms);
    CHIMERA_AVG_FIELD(phase_a_wall_ms);
    CHIMERA_AVG_FIELD(phase_a_gpu_total_ms);
    CHIMERA_AVG_FIELD(phase_a_gather_ms);
    CHIMERA_AVG_FIELD(phase_a_prefix_ms);
    CHIMERA_AVG_FIELD(phase_a_token_ids_ms);
    CHIMERA_AVG_FIELD(phase_a_d2h_ms);
    CHIMERA_AVG_FIELD(phase_a_cpu_launch_gather_ms);
    CHIMERA_AVG_FIELD(phase_a_cpu_launch_prefix_ms);
    CHIMERA_AVG_FIELD(phase_a_cpu_launch_token_ids_ms);
    CHIMERA_AVG_FIELD(phase_a_cpu_launch_d2h_ms);
    CHIMERA_AVG_FIELD(phase_a_cpu_sync_ms);
    CHIMERA_AVG_FIELD(phase_a_cpu_event_elapsed_ms);
    CHIMERA_AVG_FIELD(phase_a_gpu_sum_accounted_ms);
    CHIMERA_AVG_FIELD(phase_b_wall_ms);
    CHIMERA_AVG_FIELD(phase_b_binary_ip_total_ms);
    CHIMERA_AVG_FIELD(phase_b_doc_score_total_ms);
    CHIMERA_AVG_FIELD(phase_b_total_kernel_ms);
    CHIMERA_AVG_FIELD(phase_c_wait_d2h_ms);
    CHIMERA_AVG_FIELD(phase_c_topk_ms);
    CHIMERA_AVG_FIELD(phase_c_identify_ms);
    CHIMERA_AVG_FIELD(phase_c_gpu_extract_ms);
    CHIMERA_AVG_FIELD(phase_c_cpu_ip_ex_ms);
    CHIMERA_AVG_FIELD(phase_c_wait_extract_ms);
    CHIMERA_AVG_FIELD(phase_c_combine_ms);
    CHIMERA_AVG_FIELD(phase_c_total_ms);
    CHIMERA_AVG_FIELD(phase_c_refined_docs);
    CHIMERA_AVG_FIELD(phase_abc_total_wall_ms);
    CHIMERA_AVG_FIELD(stage2_time_ms);
    CHIMERA_AVG_FIELD(stage3_time_ms);
    CHIMERA_AVG_FIELD(stage3_docs);
    CHIMERA_AVG_FIELD(total_h2d_ms);
    CHIMERA_AVG_FIELD(total_d2h_ms);
    CHIMERA_AVG_FIELD(total_transfer_ms);
    CHIMERA_AVG_FIELD(total_h2d_bytes);
    CHIMERA_AVG_FIELD(total_d2h_bytes);
    CHIMERA_AVG_FIELD(total_transfer_bytes);
#undef CHIMERA_AVG_FIELD
}

inline void print_gpu_search_profile_v6(
    const gpu_search_profile_v6& profile,
    const char* prefix = "[PROFILE]")
{
    std::cout << prefix << ' '
              << "Mode: "
              << (profile.persistent_stage23
                      ? "Persistent Stage 2+3 (streaming top-k + system fence)"
                      : "Non-overlapping Stage 2 then 3")
              << "\n";

    if (profile.persistent_stage23) {
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
        std::cout << prefix << " Phase A (Data Preparation) wall: " << profile.phase_a_wall_ms
                  << " ms, GPU total: " << profile.phase_a_gpu_total_ms << " ms\n";
        std::cout << prefix << "   1. Gather doc lengths      : " << profile.phase_a_gather_ms
                  << " ms (CPU launch: " << profile.phase_a_cpu_launch_gather_ms << " ms)\n";
        std::cout << prefix << "   2. Prefix sum offsets      : " << profile.phase_a_prefix_ms
                  << " ms (CPU launch: " << profile.phase_a_cpu_launch_prefix_ms << " ms)\n";
        std::cout << prefix << "   3. Gather token IDs        : " << profile.phase_a_token_ids_ms
                  << " ms (CPU launch: " << profile.phase_a_cpu_launch_token_ids_ms << " ms)\n";
        std::cout << prefix << "   4. D2H metadata + sync     : " << profile.phase_a_d2h_ms
                  << " ms (CPU launch: " << profile.phase_a_cpu_launch_d2h_ms << " ms)\n";
        std::cout << prefix << "   5. cudaStreamSynchronize   : " << profile.phase_a_cpu_sync_ms << " ms\n";
        std::cout << prefix << "   6. cudaEventElapsedTime x5 : " << profile.phase_a_cpu_event_elapsed_ms << " ms\n";
        std::cout << prefix << "   GPU sum accounted           : " << profile.phase_a_gpu_sum_accounted_ms << " ms\n";
        std::cout << prefix << " Phase B wall time: " << profile.phase_b_wall_ms << " ms\n";
        std::cout << prefix
                  << " Phase C: wait_d2h=" << profile.phase_c_wait_d2h_ms
                  << " ms, topk=" << profile.phase_c_topk_ms
                  << " ms, identify=" << profile.phase_c_identify_ms
                  << " ms, gpu_extract=" << profile.phase_c_gpu_extract_ms
                  << " ms, cpu_ip_ex=" << profile.phase_c_cpu_ip_ex_ms
                  << " ms, wait_extract=" << profile.phase_c_wait_extract_ms
                  << " ms, combine=" << profile.phase_c_combine_ms
                  << " ms (total=" << profile.phase_c_total_ms
                  << " ms, " << static_cast<int>(std::llround(profile.phase_c_refined_docs))
                  << " docs)\n";
        std::cout << prefix << " Total wall time for Phase A + B + C: "
                  << profile.phase_abc_total_wall_ms << " ms\n";
        std::cout << prefix << " Phase B binary_ip total     : " << profile.phase_b_binary_ip_total_ms << " ms\n";
        std::cout << prefix << " Phase B doc_score total     : " << profile.phase_b_doc_score_total_ms << " ms\n";
        std::cout << prefix << " Phase B total kernel time   : " << profile.phase_b_total_kernel_ms << " ms\n";
        std::cout << prefix << " Total search time           : " << profile.total_search_time_ms << " ms\n";
    } else {
        std::cout << prefix << " Total GPU time: " << profile.total_search_time_ms << " ms\n";
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
        std::cout << prefix << " Stage 2 time: " << profile.stage2_time_ms << " ms\n";
        std::cout << prefix << " Stage 3 time: " << profile.stage3_time_ms
                  << " ms (" << static_cast<int>(std::llround(profile.stage3_docs)) << " docs)\n";
    }

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

struct chimera_index {
    // Scalar metadata
    size_t n;
    size_t d;
    size_t n_clusters;
    size_t ex_bits;
    size_t num_docs;

    int max_doc_len = 0;
    int max_cluster_size = 0;
    size_t workspace_probe_cluster_bound_ = 0;
    size_t workspace_probe_token_bound_ = 0;
#ifdef CHIMERA_COMPACT_DOC_BUFFER
    size_t workspace_probe_unique_doc_bound_ = 0;
#endif

    // CPU-side data
    Rotator<float>* rotator_;
    IVF_PG* ivf;
    std::vector<char> one_bit_code_;
    std::vector<char> full_code_;
    std::vector<float> one_bit_factor_;
    std::vector<float> ex_factor_;
    std::vector<int> doc_ids_;
    std::vector<int> doc_ptrs_;
    float (*ip_func_)(const float*, const uint8_t*, size_t);
    void (*unpack_func_)(const uint8_t*, float*, size_t);

    // GPU persistent data
    char*  d_one_bit_code_ = nullptr;
    float* d_one_bit_factor_ = nullptr;
    int*   d_doc_ids_ = nullptr;
    int*   d_doc_ptrs_ = nullptr;
    int*   d_doc_block_lut_ = nullptr;
    int*   d_inv_list_ = nullptr;
    gpu_cluster_pos_t* d_cluster_pos_ = nullptr;

    // Cluster-ordered copies: data reordered by inv_list so that vectors
    // in the same cluster are contiguous in memory. Stage-1 reads these
    // instead of the original arrays for coalesced global memory access.
    char*  d_clustered_code_ = nullptr;
    float* d_clustered_factor_ = nullptr;
    int*   d_clustered_doc_ids_ = nullptr;
    bool   use_clustered_ = true;  // false = fallback to non-clustered + inv_list indirection

    // Search parameters
    int nprobe = 128;
    int k_rank_cluster = 3000;
    int k_rank_all_tokens = 300;
    int itopk_size = 150;
    int overlap_chunks = 5;
    bool use_docptr_remap_ = false;

    // Pre-allocated GPU workspace
    struct Workspace {
        float* d_queries;
        float* d_cb1_sumq;

        size_t* d_emb_ids;
        int*    d_pair_offsets;
        float*  d_emb_dists;
        int*    d_pair_doc_ids;
        int*    d_pair_query_indices;

        stage2_token_id_t* d_token_ids;
        size_t* d_candidate_offsets;
        float*  d_token_dists;
        float*  d_doc_scores;

        int*    d_selected_indices;
        size_t* d_out_offsets;
        float*  d_out_one_bit_dists;

#ifdef CHIMERA_USE_LUT
        float* d_lut;
#endif

        float*        d_cagra_dists;
        uint32_t*     d_cagra_labels;

        int*    d_sorted_doc_ids;
        int*    d_sorted_query_indices;
        float*  d_sorted_dists;
        int*    d_unique_doc_ids;
        int*    d_doc_offsets;
        float*  d_stage1_doc_scores;
        float*  d_doc_query_max;
        int*    d_num_unique_docs;
#ifdef CHIMERA_COMPACT_DOC_BUFFER
        doc_bitmap_bucket_t* d_doc_bitmap;
        doc_bitmap_offset_t* d_doc_bitmap_offsets;
        size_t  doc_bitmap_bucket_count;
        size_t  doc_bitmap_offset_count;
        size_t  max_compact_docs;    // conservative estimate for unique docs
        size_t  compact_doc_capacity;
        char*   d_compact_doc_buffer_raw;
        char*   d_compact_doc_buffer;
        size_t  compact_doc_buffer_bytes;
#else
        int*    d_doc_touched;
#endif
        void*   d_cub_temp_storage;
        size_t  cub_temp_storage_bytes;

        float*  d_topk_scores;
        int*    d_topk_doc_ids;
        int*    d_topk_indices;
        size_t  topk_buf_capacity;

        float*  h_pinned_queries;
        float*  h_pinned_cb1_sumq;
        float*  h_pinned_dists;
        float*  d_pinned_dists;
        float*  h_pinned_batch_scores;

        size_t max_q_doclen;
        size_t max_stage1_pairs;
        size_t max_stage2_candidates;
        size_t max_stage2_tokens;
        size_t max_stage2_k;
        size_t max_stage2_k_tokens;
        size_t max_stage1_touched_docs;
        size_t estimated_num_docs;
        int    max_embs_per_query_bound;

        size_t* d_pst_candidate_offsets;
        stage2_token_id_t* d_pst_token_ids;

        float*        h_mapped_doc_scores;

        size_t* h_pinned_pst_candidate_offsets;
        int*    h_pinned_pst_candidate_doc_ids;
        size_t* h_pinned_pst_total_tokens;

        cudaEvent_t phase_a_start_event;
        cudaEvent_t phase_a_gather_done_event;
        cudaEvent_t phase_a_prefix_done_event;
        cudaEvent_t phase_a_token_ids_done_event;
        cudaEvent_t phase_a_d2h_done_event;

        cudaEvent_t pst_compute_done;
        cudaEvent_t pst_extract_done;
        static constexpr int PST_NUM_D2H_CHUNKS = 8;
        cudaEvent_t pst_d2h_chunk_done[PST_NUM_D2H_CHUNKS];

        cudaStream_t stream_compute;
        cudaStream_t stream_h2d;
        cudaStream_t stream_d2h;
        cudaStream_t stream_extract;

        cudaEvent_t event_h2d_done;

#ifdef CHIMERA_PROFILE
        cudaEvent_t event_start, event_end;
        cudaEvent_t event_stage1_start, event_stage1_end;
        cudaEvent_t event_stage2_start, event_stage2_end;

        cudaEvent_t s1_cagra_start, s1_cagra_end;
        cudaEvent_t s1_expansion_start, s1_expansion_end;
        cudaEvent_t s1_binary_ip_start, s1_binary_ip_end;
        cudaEvent_t s1_memset_start, s1_memset_end;
        cudaEvent_t s1_atomic_agg_start, s1_atomic_agg_end;
        cudaEvent_t s1_sum_scores_start, s1_sum_scores_end;
        cudaEvent_t s1_topk_sort_start, s1_topk_sort_end;
        cudaEvent_t s1_d2d_start, s1_d2d_end;

        cudaEvent_t s2_gather_start, s2_gather_end;
        cudaEvent_t s2_prefix_start, s2_prefix_end;
        cudaEvent_t s2_tokenids_start, s2_tokenids_end;
        cudaEvent_t s2_binaryip_start, s2_binaryip_end;
        cudaEvent_t s2_docscore_start, s2_docscore_end;
        cudaEvent_t s2_d2h_start, s2_d2h_end;
        cudaEvent_t s2_extract_start, s2_extract_end;

        float s2_gather_ms   = 0;
        float s2_prefix_ms   = 0;
        float s2_tokenids_ms = 0;
        float s2_binaryip_ms = 0;
        float s2_docscore_ms = 0;
        float s2_d2h_ms      = 0;
        float s2_topk_cpu_us = 0;
        float s2_extract_ms  = 0;
        int   s2_num_batches = 0;

        cudaEvent_t s23_pst_kernel_start, s23_pst_kernel_end;
        float s23_pst_kernel_ms   = 0;
        float s23_pst_bip_ms      = 0;
        float s23_pst_docscore_ms = 0;
        float s23_cpu_total_us   = 0;
        float s23_heap_total_us  = 0;
        int   s23_total_refined  = 0;

        static constexpr int MAX_XFER_RECORDS = 48;
        struct XferRecord {
            cudaEvent_t start, end;
            size_t bytes;
            bool is_h2d;
        };
        XferRecord xfer_records[MAX_XFER_RECORDS];
        int xfer_count = 0;
#endif
        std::vector<int>                   running_indices;
        std::vector<int>                   h_sel_indices;
        std::vector<size_t>                h_out_offsets;
        std::vector<float>                 ip_ex_buf;
        std::vector<std::pair<float, int>> refined_scores;
        std::vector<bool>                  seen_doc_map;
    } ws_{};

    // Constructor / destructor
    chimera_index(
        const std::string& filename,
        const std::vector<int>& doc_lens,
        const gpu_search_runtime_options& runtime_options);
    ~chimera_index();

    void set_doc_mapping(const std::vector<int>& doc_lens);
    void compute_workspace_probe_bounds();
    void allocate_workspace();
#ifdef CHIMERA_COMPACT_DOC_BUFFER
    void ensure_compact_doc_capacity(size_t required_rows);
    void ensure_stage1_sort_capacity(size_t required_rows);
#endif
    size_t doc_len(size_t doc_id) const;

    std::vector<size_t> search(const float* queries, size_t k);
    std::vector<size_t> search_profiled(const float* queries, size_t k);
    std::vector<size_t> search_profiled(
        const float* queries,
        size_t k,
        gpu_search_profile_v6* profile,
        bool print_profile = true);
    template <bool kProfile>
    std::vector<size_t> search_impl(
        const float* queries,
        size_t k,
        gpu_search_profile_v6* profile = nullptr,
        bool print_profile = true);

    void rank_cluster_dists_gpu(
        query_object* h_query_objs,
        size_t nprobe, size_t k,
        int& actual_k_out,
        cudaStream_t stream = 0);
    template <bool kProfile>
    void rank_cluster_dists_gpu_impl(
        query_object* h_query_objs,
        size_t nprobe, size_t k,
        int& actual_k_out,
        cudaStream_t stream = 0);

    static constexpr int N_OVERLAP_CHUNKS = 64;

    void rank_stage23_persistent(
        int num_candidates,
        size_t k,
        size_t k_stage2,
        query_object* queries,
        std::vector<size_t>& result);
    template <bool kProfile>
    void rank_stage23_persistent_impl(
        int num_candidates,
        size_t k,
        size_t k_stage2,
        query_object* queries,
        std::vector<size_t>& result,
        gpu_search_profile_v6* profile = nullptr,
        bool print_profile = true);

    void cpu_compute_ip_ex(
        const int* h_candidate_doc_ids,
        const int* h_sel_indices,
        const size_t* h_out_offsets,
        int to_refine,
        const float* queries_flat,
        float* ip_ex_buf);

    void cpu_combine_scores(
        const int* h_candidate_doc_ids,
        const int* h_sel_indices,
        const size_t* h_out_offsets,
        int to_refine,
        const query_object* queries,
        const float* ip_full_buf,
        std::pair<float, int>* refined_scores);
};


}  // namespace Chimera
