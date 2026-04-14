#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "gpu_search_options.hpp"
#include "gpu_config.cuh"
#include "ivf_pg.hpp"

using pg_label_t = uint32_t;

namespace rabitqlib {
template <typename T>
class Rotator;
}

struct query_object;

struct cast_int_size_t {
    __host__ __device__ size_t operator()(int x) const { return (size_t)x; }
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
    rabitqlib::Rotator<float>* rotator_ = nullptr;
    IVF_PG* ivf = nullptr;
    std::vector<char> one_bit_code_;
    std::vector<char> ex_code_;
    std::vector<float> one_bit_factor_;
    std::vector<float> ex_factor_;
    std::vector<int> doc_ids_;
    std::vector<int> doc_ptrs_;
    float (*ip_func_)(const float*, const uint8_t*, size_t);
    void (*unpack_func_)(const uint8_t*, float*, size_t);

    // GPU persistent data
    char* d_one_bit_code_ = nullptr;
    float* d_one_bit_factor_ = nullptr;
    int* d_doc_ids_ = nullptr;
    int* d_doc_ptrs_ = nullptr;
    int* d_inv_list_ = nullptr;
    size_t* d_cluster_pos_ = nullptr;

    // Search parameters
    int nprobe = 128;
    int k_rank_cluster = 1800;
    int k_rank_all_tokens = 300;
    int itopk_size = 150;
    int overlap_chunks = 5;

    struct Workspace {
        float* d_queries;
        float* d_cb1_sumq;

        size_t* d_emb_ids;
        int* d_pair_offsets;
        float* d_emb_dists;
        int* d_pair_doc_ids;
        int* d_pair_query_indices;

        size_t* d_token_ids;
        size_t* d_candidate_offsets;
        float* d_token_dists;
        float* d_doc_scores;
        float* d_stage2_doc_query_max;

        int* d_selected_indices;
        size_t* d_out_offsets;
        float* d_out_one_bit_dists;

#if GPU_MVR_BINARY_IP_IMPL == 2
        float* d_lut;
#endif

        float* d_cagra_dists;
        pg_label_t* d_cagra_labels;

        int* d_sorted_doc_ids;
        int* d_sorted_query_indices;
        float* d_sorted_dists;
        int* d_unique_doc_ids;
        int* d_doc_offsets;
        float* d_stage1_doc_scores;
        float* d_doc_query_max;
        int* d_num_unique_docs;
        int* d_doc_touched;
        void* d_cub_temp_storage;
        size_t cub_temp_storage_bytes;

        float* d_topk_scores;
        int* d_topk_doc_ids;
        int* d_topk_indices;

        float* h_pinned_queries;
        float* h_pinned_cb1_sumq;
        float* h_pinned_dists;
        float* h_pinned_batch_scores;

        size_t max_q_doclen;
        size_t max_stage1_pairs;
        size_t max_stage2_candidates;
        size_t max_stage2_tokens;
        size_t max_stage2_k;
        size_t max_stage2_k_tokens;
        size_t estimated_num_docs;
        int max_embs_per_query_bound;

        size_t* d_pst_candidate_offsets;
        size_t* d_pst_token_ids;
        float* h_mapped_doc_scores;
        std::vector<size_t> v_pst_candidate_offsets;

        cudaEvent_t pst_compute_done;
        cudaEvent_t pst_extract_done;
        static constexpr int PST_NUM_D2H_CHUNKS = 8;
        cudaEvent_t pst_d2h_chunk_done[PST_NUM_D2H_CHUNKS];

        cudaStream_t stream_compute;
        cudaStream_t stream_h2d;
        cudaStream_t stream_d2h;

        cudaEvent_t event_h2d_done;
#if GPU_MVR_STAGE1_SPARSE_CLEANUP == 1
        cudaEvent_t event_stage1_sum_done;
        cudaEvent_t event_stage1_cleanup_done;
#endif

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

        float s2_gather_ms = 0;
        float s2_prefix_ms = 0;
        float s2_tokenids_ms = 0;
        float s2_binaryip_ms = 0;
        float s2_docscore_ms = 0;
        float s2_d2h_ms = 0;
        float s2_topk_cpu_us = 0;
        float s2_extract_ms = 0;
        int s2_num_batches = 0;

        cudaEvent_t s23_pst_kernel_start, s23_pst_kernel_end;
        float s23_pst_kernel_ms = 0;
        float s23_cpu_total_us = 0;
        float s23_heap_total_us = 0;
        int s23_total_refined = 0;

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
    } ws_;

    gpu_mvr_index(
        const std::string& filename,
        const std::vector<int>& doc_lens,
        const gpu_search_runtime_options& runtime_options);

    void set_doc_mapping(const std::vector<int>& doc_lens);
    void allocate_workspace();
    size_t doc_len(size_t doc_id) const;

    std::vector<size_t> search(const float* queries, size_t k);

    void rank_cluster_dists_gpu(
        query_object* h_query_objs,
        size_t nprobe,
        size_t k,
        int& actual_k_out,
        cudaStream_t stream = 0);

    void rank_all_tokens_1bit_gpu(
        int num_candidates,
        size_t k,
        std::vector<size_t>& output_ids,
        std::vector<float>& one_bit_dists,
        cudaStream_t stream = 0);

    void rank_stage23_persistent(
        int num_candidates,
        size_t k,
        size_t k_stage2,
        query_object* queries,
        std::vector<size_t>& result);

    void rank_all_tokens_exbits_cpu(
        query_object* queries,
        std::vector<size_t>& input_ids,
        std::vector<float>& one_bit_dists,
        size_t k,
        std::vector<size_t>& output_ids);

    ~gpu_mvr_index();
};
