#pragma once

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
#include <cfloat>
#include <limits>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <immintrin.h>
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "quantization.hpp"
#include "estimator.hpp"
#include "query.hpp"
#include "ivf_pg.hpp"
#include "gpu_config.cuh"
#include "gpu_kernels.cuh"

using namespace rabitqlib;

#ifndef GPU_MVR_CUDA_CHECK_DEFINED
#define GPU_MVR_CUDA_CHECK_DEFINED
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// === Transfer profiling macros ===
#ifdef GPU_MVR_PROFILE
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

struct gpu_mvr_index {
    // Scalar metadata
    size_t n;
    size_t d;
    size_t n_clusters;
    size_t ex_bits;
    size_t num_docs;

    int max_doc_len = 0;
    int max_cluster_size = 0;

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
    char*  d_one_bit_code_;
    float* d_one_bit_factor_;
    int*   d_doc_ids_;
    int*   d_doc_ptrs_;
    int*   d_inv_list_;
    size_t* d_cluster_pos_;

    // Cluster-ordered copies: data reordered by inv_list so that vectors
    // in the same cluster are contiguous in memory. Stage-1 reads these
    // instead of the original arrays for coalesced global memory access.
    char*  d_clustered_code_;
    float* d_clustered_factor_;
    int*   d_clustered_doc_ids_;
    bool   use_clustered_ = true;  // false = fallback to non-clustered + inv_list indirection

    // Search parameters
    int nprobe = 128;
    int k_rank_cluster = 3000;
    int k_rank_all_tokens = 300;

    // Pre-allocated GPU workspace
    struct Workspace {
        float* d_queries;
        float* d_cb1_sumq;

        size_t* d_emb_ids;
        int*    d_pair_offsets;
        float*  d_emb_dists;
        int*    d_pair_doc_ids;
        int*    d_pair_query_indices;

        size_t* d_token_ids;
        size_t* d_candidate_offsets;
        float*  d_token_dists;
        float*  d_doc_scores;

        int*    d_selected_indices;
        size_t* d_out_offsets;
        float*  d_out_one_bit_dists;

#ifdef GPU_MVR_USE_LUT
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
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
        // Hash table for doc_id → compact_slot mapping (replaces d_doc_touched)
        int*    d_ht_keys;           // [ht_capacity], init -1
        int*    d_ht_vals;           // [ht_capacity], init -1
        size_t  ht_capacity;         // power of 2
        unsigned int ht_mask;        // ht_capacity - 1
        size_t  max_compact_docs;    // max unique docs (d_doc_query_max rows)
#else
        int*    d_doc_touched;
#endif
        void*   d_cub_temp_storage;
        size_t  cub_temp_storage_bytes;

        float*  d_topk_scores;
        int*    d_topk_doc_ids;
        int*    d_topk_indices;

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
        size_t estimated_num_docs;
        int    max_embs_per_query_bound;

        size_t* d_pst_candidate_offsets;
        size_t* d_pst_token_ids;

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

#ifdef GPU_MVR_PROFILE
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
    } ws_;

    // Constructor / destructor
    gpu_mvr_index(const std::string& filename, const std::vector<int>& doc_lens);
    ~gpu_mvr_index();

    void set_doc_mapping(const std::vector<int>& doc_lens);
    void allocate_workspace();
    size_t doc_len(size_t doc_id) const;

    std::vector<size_t> search(const float* queries, size_t k);

    void rank_cluster_dists_gpu(
        query_object* h_query_objs,
        size_t nprobe, size_t k,
        int& actual_k_out,
        cudaStream_t stream = 0);

    static constexpr int N_OVERLAP_CHUNKS = 5;
    int PRELIM_PER_CHUNK = k_rank_all_tokens / N_OVERLAP_CHUNKS;

    void rank_stage23_persistent(
        int num_candidates,
        size_t k,
        size_t k_stage2,
        query_object* queries,
        std::vector<size_t>& result);

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
