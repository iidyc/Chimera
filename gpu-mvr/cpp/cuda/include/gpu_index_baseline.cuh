#pragma once

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/iterator/transform_iterator.h>
#include <cub/cub.cuh>

#define EIGEN_NO_CUDA

#include <vector>
#include <queue>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cfloat>
#include <limits>
#include <fstream>
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "quantization.hpp"
#include "estimator.hpp"
#include "query.hpp"
#include "ivf_pg.hpp"
#include "gpu_config.cuh"
#include "gpu_kernels_baseline.cuh"

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

struct cast_int_size_t_baseline {
    __host__ __device__ size_t operator()(int x) const { return (size_t)x; };
};

struct gpu_mvr_index_baseline {
    size_t n;
    size_t d;
    size_t n_clusters;
    size_t ex_bits;
    size_t num_docs;

    int max_doc_len = 0;
    int max_cluster_size = 0;

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

    char*   d_one_bit_code_;
    float*  d_one_bit_factor_;
    int*    d_doc_ids_;
    int*    d_doc_ptrs_;
    int*    d_inv_list_;
    size_t* d_cluster_pos_;

    int nprobe = 128;
    int k_rank_cluster = 5000;
    int k_rank_all_tokens = 300;

    struct Workspace {
        float* d_queries;
        float* d_cb1_sumq;

        size_t* d_emb_ids;
        int*    d_pair_offsets;
        float*  d_emb_dists;
        int*    d_pair_doc_ids;

        float*  d_doc_query_max;
        float*  d_stage1_doc_scores;

        float*  d_topk_scores;
        int*    d_topk_doc_ids;
        int*    d_topk_indices;
        int*    d_sorted_doc_ids;

        void*   d_cub_temp_storage;
        size_t  cub_temp_storage_bytes;

        size_t* d_token_ids;
        size_t* d_candidate_offsets;
        float*  d_token_dists;
        float*  d_doc_scores;

        int*    d_selected_indices;
        size_t* d_out_offsets;
        float*  d_out_one_bit_dists;

        float*        d_cagra_dists;
        uint32_t*     d_cagra_labels;

        float*  h_pinned_queries;
        float*  h_pinned_cb1_sumq;
        float*  h_pinned_dists;
        int*    h_pinned_candidate_doc_ids;

        cudaStream_t stream_compute;
        cudaStream_t stream_h2d;
        cudaStream_t stream_d2h;
        cudaEvent_t  event_h2d_done;

        size_t max_q_doclen;
        size_t max_stage1_pairs;
        size_t max_stage2_candidates;
        size_t max_stage2_tokens;
        size_t max_stage2_k;
        size_t max_stage2_k_tokens;
        size_t estimated_num_docs;
        int    max_embs_per_query_bound;
    } ws_;

    gpu_mvr_index_baseline(const std::string& filename, const std::vector<int>& doc_lens);
    ~gpu_mvr_index_baseline();

    void set_doc_mapping(const std::vector<int>& doc_lens);
    void allocate_workspace();
    size_t doc_len(size_t doc_id) const;

    std::vector<size_t> search(const float* queries, size_t k);

    void rank_cluster_dists_baseline(
        query_object* h_query_objs,
        size_t nprobe, size_t k,
        int& actual_k_out,
        cudaStream_t stream = 0);

    void rank_all_tokens_1bit_baseline(
        int num_candidates,
        size_t k,
        std::vector<size_t>& output_ids,
        std::vector<float>& one_bit_dists,
        cudaStream_t stream = 0);

    void rank_all_tokens_exbits_cpu(
        query_object* queries,
        std::vector<size_t>& input_ids,
        std::vector<float>& one_bit_dists,
        size_t k,
        std::vector<size_t>& output_ids);
};
