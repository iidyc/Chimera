#pragma once

// Baseline (unoptimized) GPU search pipeline for performance comparison.
// Produces identical search results to the optimized gpu_mvr_index in gpu_index.cuh.
// Uses the same CUDA streams, pinned memory, and async transfer infrastructure.
// Only kernel implementations differ: all compute kernels are naive/unoptimized.
//
// Differences from the optimized version:
//   - No shared memory in binary IP, doc scoring, or extract kernels
//   - No LUT-based binary IP path
//   - No sparse document tracking (full-matrix memset + sum over all docs)
//   - No GPU_MVR_OVERLAP_STAGE23 (sequential Stage 2 → Stage 3 only)
//   - No profiling instrumentation
//   - Naive byte-by-byte binary code access (no vectorized uint64_t loads)
//   - 2D grid for Stage 2 binary IP (one thread per query×token pair)
//   - Single-thread doc scoring (no tiling, no block reduction)

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
    std::vector<char> ex_code_;
    std::vector<float> one_bit_factor_;
    std::vector<float> ex_factor_;
    std::vector<int> doc_ids_;
    std::vector<int> doc_ptrs_;
    float (*ip_func_)(const float*, const uint8_t*, size_t);
    void (*unpack_func_)(const uint8_t*, float*, size_t);

    // GPU persistent data
    char*   d_one_bit_code_;
    float*  d_one_bit_factor_;
    int*    d_doc_ids_;
    int*    d_doc_ptrs_;
    int*    d_inv_list_;
    size_t* d_cluster_pos_;

    // Search parameters
    int nprobe = 128;
    int k_rank_cluster = 5000;
    int k_rank_all_tokens = 300;

    // Simplified workspace (no sparse tracking, no LUT, no overlap, no profiling)
    struct Workspace {
        // Query data
        float* d_queries;
        float* d_cb1_sumq;

        // Stage 1 workspace
        size_t* d_emb_ids;
        int*    d_pair_offsets;
        float*  d_emb_dists;
        int*    d_pair_doc_ids;   // reused as doc_lengths buffer in Stage 2

        // Stage 1 aggregation (full matrix, no sparse tracking)
        float*  d_doc_query_max;     // [Q_DOCLEN * num_docs]
        float*  d_stage1_doc_scores; // [num_docs]

        // Top-k workspace
        float*  d_topk_scores;
        int*    d_topk_doc_ids;
        int*    d_topk_indices;
        int*    d_sorted_doc_ids;    // CUB sort output

        // CUB temporary storage
        void*   d_cub_temp_storage;
        size_t  cub_temp_storage_bytes;

        // Stage 2 workspace
        size_t* d_token_ids;
        size_t* d_candidate_offsets;
        float*  d_token_dists;
        float*  d_doc_scores;

        // Extract workspace
        int*    d_selected_indices;
        size_t* d_out_offsets;
        float*  d_out_one_bit_dists;

        // CAGRA batched search workspace
        float*        d_cagra_dists;
        faiss::idx_t* d_cagra_labels;

        // Pinned host memory (same async infrastructure as optimized)
        float*  h_pinned_queries;
        float*  h_pinned_cb1_sumq;
        float*  h_pinned_dists;
        int*    h_pinned_candidate_doc_ids;

        // CUDA streams and events
        cudaStream_t stream_compute;
        cudaStream_t stream_h2d;
        cudaStream_t stream_d2h;
        cudaEvent_t  event_h2d_done;

        // Sizing parameters
        size_t max_q_doclen;
        size_t max_stage1_pairs;
        size_t max_stage2_candidates;
        size_t max_stage2_tokens;
        size_t max_stage2_k;
        size_t max_stage2_k_tokens;
        size_t estimated_num_docs;
        int    max_embs_per_query_bound;
    } ws_;

    // Construct from file (same as optimized)
    gpu_mvr_index_baseline(const std::string& filename, const std::vector<int>& doc_lens) {
        std::ifstream inf(filename, std::ios::binary);
        inf.read((char*)&n, sizeof(size_t));
        inf.read((char*)&d, sizeof(size_t));
        inf.read((char*)&n_clusters, sizeof(size_t));
        inf.read((char*)&ex_bits, sizeof(size_t));

        size_t file_padded_dim;
        inf.read((char*)&file_padded_dim, sizeof(size_t));

        if (file_padded_dim != PADDED_DIM) {
            inf.close();
            throw std::runtime_error(
                "Index file padded_dim=" + std::to_string(file_padded_dim) +
                " does not match compiled PADDED_DIM=" + std::to_string(PADDED_DIM)
            );
        }

        one_bit_code_.resize(n * PADDED_DIM / 8);
        ex_code_.resize(n * PADDED_DIM * ex_bits / 8);
        one_bit_factor_.resize(n);
        ex_factor_.resize(n);

        inf.read(one_bit_code_.data(), one_bit_code_.size());
        inf.read(ex_code_.data(), ex_code_.size());
        inf.read((char*)one_bit_factor_.data(), n * sizeof(float));
        inf.read((char*)ex_factor_.data(), n * sizeof(float));
        inf.close();

        rotator_ = choose_rotator<float>(d, RotatorType::FhtKacRotator, PADDED_DIM);
        std::ifstream rot_in("rotator.bin", std::ios::binary);
        rotator_->load(rot_in);
        rot_in.close();

        ip_func_ = select_excode_ipfunc(ex_bits);
        unpack_func_ = select_excode_unpackfunc(ex_bits);

        ivf = new IVF_PG(n_clusters, d, PGType::CAGRA);
        ivf->load(filename);
        max_cluster_size = ivf->max_cluster_size();

        set_doc_mapping(doc_lens);

        // Allocate persistent GPU data
        size_t code_bytes = n * PADDED_DIM / 8;
        CUDA_CHECK(cudaMalloc(&d_one_bit_code_, code_bytes));
        CUDA_CHECK(cudaMalloc(&d_one_bit_factor_, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_doc_ids_, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_doc_ptrs_, (num_docs + 1) * sizeof(int)));

        size_t inv_list_size = ivf->inv_list.size();
        CUDA_CHECK(cudaMalloc(&d_inv_list_, inv_list_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cluster_pos_, (ivf->n_clusters + 1) * sizeof(size_t)));

        CUDA_CHECK(cudaMemcpy(d_inv_list_, ivf->inv_list.data(),
                              inv_list_size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cluster_pos_, ivf->cluster_pos.data(),
                              (ivf->n_clusters + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_one_bit_code_, one_bit_code_.data(), code_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_one_bit_factor_, one_bit_factor_.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_doc_ids_, doc_ids_.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_doc_ptrs_, doc_ptrs_.data(), (num_docs + 1) * sizeof(int), cudaMemcpyHostToDevice));

        allocate_workspace();
    }

    inline void set_doc_mapping(const std::vector<int>& doc_lens) {
        num_docs = doc_lens.size();
        doc_ptrs_.resize(num_docs + 1, 0);
        for (size_t i = 0; i < num_docs; ++i) {
            max_doc_len = std::max(max_doc_len, doc_lens[i]);
            doc_ptrs_[i + 1] = doc_ptrs_[i] + doc_lens[i];
        }
        doc_ids_.resize(n);
        for (size_t i = 0; i < num_docs; ++i) {
            for (size_t j = 0; j < (size_t)doc_lens[i]; ++j) {
                doc_ids_[doc_ptrs_[i] + j] = i;
            }
        }
    }

    inline void allocate_workspace() {
        ws_.max_q_doclen = Q_DOCLEN;
        ws_.max_stage1_pairs = (size_t)nprobe * max_cluster_size * Q_DOCLEN;
        ws_.max_stage2_candidates = k_rank_cluster;
        ws_.max_stage2_tokens = ws_.max_stage2_candidates * max_doc_len;
        ws_.max_stage2_k = k_rank_all_tokens;
        ws_.max_stage2_k_tokens = ws_.max_stage2_k * max_doc_len;
        ws_.estimated_num_docs = (size_t)num_docs;
        ws_.max_embs_per_query_bound = nprobe * max_cluster_size;

        // Query workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_cb1_sumq, Q_DOCLEN * sizeof(float)));

        // Stage 1 workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_emb_ids, ws_.max_stage1_pairs * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_pair_offsets, (Q_DOCLEN + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_emb_dists, ws_.max_stage1_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_pair_doc_ids, ws_.max_stage1_pairs * sizeof(int)));

        // Stage 1 aggregation: full matrix (no sparse tracking)
        CUDA_CHECK(cudaMalloc(&ws_.d_doc_query_max, ws_.estimated_num_docs * Q_DOCLEN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_stage1_doc_scores, ws_.estimated_num_docs * sizeof(float)));

        // Top-k workspace (must fit num_docs for Stage 1 sort over all docs)
        size_t topk_buf_size = std::max((size_t)num_docs, ws_.max_stage2_candidates);
        CUDA_CHECK(cudaMalloc(&ws_.d_topk_scores, topk_buf_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_topk_doc_ids, topk_buf_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_topk_indices, topk_buf_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_sorted_doc_ids, topk_buf_size * sizeof(int)));

        // CUB temporary storage — query sizes for all operations
        ws_.d_cub_temp_storage = nullptr;
        ws_.cub_temp_storage_bytes = 0;
        size_t temp_bytes = 0;

        // Stage 1: SortPairsDescending for doc scores (over ALL num_docs)
        temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_bytes,
            ws_.d_stage1_doc_scores, ws_.d_topk_scores,
            ws_.d_topk_doc_ids, ws_.d_sorted_doc_ids,
            (int)num_docs, 0, 32
        );
        ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);

        // Stage 1: InclusiveScan for pair offsets
        temp_bytes = 0;
        cub::DeviceScan::InclusiveScan(
            nullptr, temp_bytes,
            ws_.d_pair_offsets + 1, ws_.d_pair_offsets + 1,
            thrust::plus<int>(), (int)Q_DOCLEN
        );
        ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);

        // Stage 2 workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_token_ids, ws_.max_stage2_tokens * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_candidate_offsets, (ws_.max_stage2_candidates + 1) * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_token_dists, ws_.max_stage2_tokens * Q_DOCLEN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_doc_scores, ws_.max_stage2_candidates * sizeof(float)));

        // Extract workspace (allocated before CUB dry runs that reference d_selected_indices)
        CUDA_CHECK(cudaMalloc(&ws_.d_selected_indices, ws_.max_stage2_k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_out_offsets, (ws_.max_stage2_k + 1) * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_out_one_bit_dists, ws_.max_stage2_k_tokens * Q_DOCLEN * sizeof(float)));

        // Stage 2: SortPairsDescending for candidate scores
        temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_bytes,
            ws_.d_doc_scores, ws_.d_topk_scores,
            ws_.d_selected_indices, ws_.d_topk_indices,
            (int)ws_.max_stage2_candidates, 0, 32
        );
        ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);

        // Stage 2: InclusiveScan for candidate offsets (int -> size_t with transform)
        {
            thrust::device_ptr<int> len_ptr(ws_.d_pair_doc_ids);
            auto xform_iter = thrust::make_transform_iterator(len_ptr, cast_int_size_t_baseline());
            temp_bytes = 0;
            cub::DeviceScan::InclusiveScan(
                nullptr, temp_bytes,
                xform_iter, ws_.d_candidate_offsets + 1,
                thrust::plus<size_t>(), (int)ws_.max_stage2_candidates
            );
            ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);
        }

        CUDA_CHECK(cudaMalloc(&ws_.d_cub_temp_storage, ws_.cub_temp_storage_bytes));

        // CAGRA workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_cagra_dists, Q_DOCLEN * nprobe * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_cagra_labels, Q_DOCLEN * nprobe * sizeof(faiss::idx_t)));

        // Pinned host memory (same async infrastructure as optimized)
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_cb1_sumq, Q_DOCLEN * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_dists, ws_.max_stage2_k_tokens * Q_DOCLEN * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_candidate_doc_ids, ws_.max_stage2_candidates * sizeof(int)));

        // CUDA streams and events
        CUDA_CHECK(cudaStreamCreate(&ws_.stream_compute));
        CUDA_CHECK(cudaStreamCreate(&ws_.stream_h2d));
        CUDA_CHECK(cudaStreamCreate(&ws_.stream_d2h));
        CUDA_CHECK(cudaEventCreate(&ws_.event_h2d_done));
    }

    inline size_t doc_len(size_t doc_id) const {
        return doc_ptrs_[doc_id + 1] - doc_ptrs_[doc_id];
    }

    // ======================== SEARCH PIPELINE ========================

    inline std::vector<size_t> search(const float* queries, size_t k) {
        // Step 1: Rotate queries on CPU into pinned memory
        for (size_t i = 0; i < Q_DOCLEN; ++i) {
            rotator_->rotate(&queries[i * d], &ws_.h_pinned_queries[i * PADDED_DIM]);
        }

        // Build query objects on CPU
        std::vector<query_object> query_objs(Q_DOCLEN);
        for (size_t i = 0; i < Q_DOCLEN; ++i) {
            query_objs[i] = query_object(&ws_.h_pinned_queries[i * PADDED_DIM], PADDED_DIM, ex_bits);
            ws_.h_pinned_cb1_sumq[i] = query_objs[i].cb1_sumq;
        }

        // Async H2D transfer (same streams as optimized)
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_queries, ws_.h_pinned_queries,
                                   Q_DOCLEN * PADDED_DIM * sizeof(float),
                                   cudaMemcpyHostToDevice, ws_.stream_h2d));
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_cb1_sumq, ws_.h_pinned_cb1_sumq,
                                   Q_DOCLEN * sizeof(float),
                                   cudaMemcpyHostToDevice, ws_.stream_h2d));

        CUDA_CHECK(cudaEventRecord(ws_.event_h2d_done, ws_.stream_h2d));
        CUDA_CHECK(cudaStreamWaitEvent(ws_.stream_compute, ws_.event_h2d_done));

        // Stage 1: GPU-only aggregation and top-k
        int actual_k_stage1 = 0;
        rank_cluster_dists_baseline(query_objs.data(), nprobe, k_rank_cluster,
                                     actual_k_stage1, ws_.stream_compute);

        // Stage 2: GPU 1-bit scoring
        std::vector<size_t> stage2_doc_ids;
        std::vector<float>  stage2_one_bit_dists;

        rank_all_tokens_1bit_baseline(actual_k_stage1, k_rank_all_tokens,
                                       stage2_doc_ids, stage2_one_bit_dists,
                                       ws_.stream_compute);

        // Stage 3: CPU ex-bits refinement (identical to optimized)
        std::vector<size_t> result;
        rank_all_tokens_exbits_cpu(query_objs.data(),
                                   stage2_doc_ids, stage2_one_bit_dists,
                                   k, result);

        return result;
    }

    // ======================== STAGE 1: GPU ========================

    inline void rank_cluster_dists_baseline(
        query_object* h_query_objs,
        size_t nprobe, size_t k,
        int& actual_k_out,
        cudaStream_t stream = 0
    ) {
        // Full-matrix memset (no sparse tracking)
        size_t matrix_size = (size_t)num_docs * Q_DOCLEN;
        CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_query_max, 0, matrix_size * sizeof(float), stream));

        // 1. Batched CAGRA search
        ivf->search_batch_gpu(ws_.d_queries, Q_DOCLEN, nprobe,
                              ws_.d_cagra_dists, ws_.d_cagra_labels);

        // 2. GPU cluster expansion
        compute_query_expansion_sizes_baseline_kernel<<<(Q_DOCLEN + 255) / 256, 256, 0, stream>>>(
            ws_.d_cagra_labels, d_cluster_pos_,
            ws_.d_pair_offsets + 1,
            nprobe, ivf->n_clusters, Q_DOCLEN
        );

        CUDA_CHECK(cudaMemsetAsync(ws_.d_pair_offsets, 0, sizeof(int), stream));

        {
            size_t scan_temp = ws_.cub_temp_storage_bytes;
            cub::DeviceScan::InclusiveScan(
                ws_.d_cub_temp_storage, scan_temp,
                ws_.d_pair_offsets + 1, ws_.d_pair_offsets + 1,
                thrust::plus<int>(), (int)Q_DOCLEN, stream);
        }

        int total_pairs;
        CUDA_CHECK(cudaMemcpyAsync(&total_pairs, ws_.d_pair_offsets + Q_DOCLEN,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (total_pairs == 0) {
            actual_k_out = 0;
            return;
        }

        expand_cluster_ids_baseline_kernel<<<Q_DOCLEN, 256, 0, stream>>>(
            ws_.d_cagra_labels, d_inv_list_, d_cluster_pos_,
            ws_.d_pair_offsets, ws_.d_emb_ids,
            nprobe, ivf->n_clusters, Q_DOCLEN
        );

        // 3. Naive binary IP (no shared memory, byte-by-byte)
        int max_embs_per_query = ws_.max_embs_per_query_bound;
        int threads_per_block = 256;
        int blocks_x = (max_embs_per_query + threads_per_block - 1) / threads_per_block;
        dim3 grid(blocks_x, Q_DOCLEN);

        stage1_binary_ip_baseline_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_queries, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
            ws_.d_emb_ids, ws_.d_pair_offsets, ws_.d_emb_dists,
            max_embs_per_query
        );
        CUDA_CHECK(cudaGetLastError());

        // 4. Naive aggregation (no sparse doc tracking)
        int agg_threads = 256;
        int agg_blocks = (total_pairs + agg_threads - 1) / agg_threads;

        aggregate_stage1_baseline_kernel<<<agg_blocks, agg_threads, 0, stream>>>(
            ws_.d_emb_ids, ws_.d_emb_dists, ws_.d_pair_offsets, d_doc_ids_,
            ws_.d_doc_query_max,
            num_docs, total_pairs, max_embs_per_query
        );
        CUDA_CHECK(cudaGetLastError());

        // 5. Sum scores over ALL documents (not sparse)
        int sum_threads = 256;
        int sum_blocks = ((int)num_docs + sum_threads - 1) / sum_threads;

        sum_doc_scores_baseline_kernel<<<sum_blocks, sum_threads, 0, stream>>>(
            ws_.d_doc_query_max,
            ws_.d_stage1_doc_scores,
            ws_.d_topk_doc_ids,
            num_docs
        );
        CUDA_CHECK(cudaGetLastError());

        // 6. Top-k: sort ALL docs by score descending
        cub::DeviceRadixSort::SortPairsDescending(
            ws_.d_cub_temp_storage, ws_.cub_temp_storage_bytes,
            ws_.d_stage1_doc_scores, ws_.d_topk_scores,
            ws_.d_topk_doc_ids, ws_.d_sorted_doc_ids,
            (int)num_docs, 0, 32, stream
        );
        CUDA_CHECK(cudaGetLastError());

        actual_k_out = std::min((int)k, (int)num_docs);

        // Copy top-k doc IDs to d_topk_doc_ids for Stage 2
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_topk_doc_ids, ws_.d_sorted_doc_ids,
                                   actual_k_out * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    }

    // ======================== STAGE 2: GPU ========================

    inline void rank_all_tokens_1bit_baseline(
        int num_candidates,
        size_t k,
        std::vector<size_t>& output_ids,
        std::vector<float>& one_bit_dists,
        cudaStream_t stream = 0
    ) {
        if (num_candidates == 0) return;

        // 1. Gather document lengths
        int threads = 256;
        int blocks = (num_candidates + threads - 1) / threads;

        gather_doc_lengths_baseline_kernel<<<blocks, threads, 0, stream>>>(
            ws_.d_topk_doc_ids, d_doc_ptrs_,
            ws_.d_pair_doc_ids,  // reuse as doc_lengths buffer
            num_candidates
        );
        CUDA_CHECK(cudaGetLastError());

        // 2. Prefix sum for candidate offsets
        CUDA_CHECK(cudaMemsetAsync(ws_.d_candidate_offsets, 0, sizeof(size_t), stream));
        {
            thrust::device_ptr<int> len_ptr(ws_.d_pair_doc_ids);
            auto xform_iter = thrust::make_transform_iterator(len_ptr, cast_int_size_t_baseline());
            size_t scan_temp = ws_.cub_temp_storage_bytes;
            cub::DeviceScan::InclusiveScan(
                ws_.d_cub_temp_storage, scan_temp,
                xform_iter, ws_.d_candidate_offsets + 1,
                thrust::plus<size_t>(), num_candidates, stream);
        }

        size_t total_tokens;
        CUDA_CHECK(cudaMemcpyAsync(&total_tokens, ws_.d_candidate_offsets + num_candidates,
                                   sizeof(size_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 3. Build token IDs on GPU
        gather_token_ids_baseline_kernel<<<num_candidates, 256, 0, stream>>>(
            ws_.d_topk_doc_ids, d_doc_ptrs_,
            ws_.d_candidate_offsets, ws_.d_token_ids,
            num_candidates
        );
        CUDA_CHECK(cudaGetLastError());

        // 4. Naive binary IP: 2D grid (tokens × queries), no shared memory
        int bip_threads = 256;
        int bip_blocks_x = (total_tokens + bip_threads - 1) / bip_threads;
        dim3 bip_grid(bip_blocks_x, Q_DOCLEN);

        stage2_binary_ip_baseline_kernel<<<bip_grid, bip_threads, 0, stream>>>(
            ws_.d_queries, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
            ws_.d_token_ids, ws_.d_token_dists,
            total_tokens, total_tokens
        );
        CUDA_CHECK(cudaGetLastError());

        // 5. Naive doc scoring: one thread per candidate
        int score_threads = 256;
        int score_blocks = (num_candidates + score_threads - 1) / score_threads;

        doc_score_baseline_kernel<<<score_blocks, score_threads, 0, stream>>>(
            ws_.d_token_dists, ws_.d_candidate_offsets, ws_.d_doc_scores,
            total_tokens, num_candidates
        );
        CUDA_CHECK(cudaGetLastError());

        // 6. Top-k via CUB sort
        thrust::device_ptr<int> indices_ptr(ws_.d_selected_indices);
        thrust::sequence(thrust::cuda::par.on(stream),
                        indices_ptr, indices_ptr + num_candidates);

        cub::DeviceRadixSort::SortPairsDescending(
            ws_.d_cub_temp_storage, ws_.cub_temp_storage_bytes,
            ws_.d_doc_scores, ws_.d_topk_scores,
            ws_.d_selected_indices, ws_.d_topk_indices,
            num_candidates, 0, 32, stream
        );
        CUDA_CHECK(cudaGetLastError());

        size_t actual_k = std::min(k, (size_t)num_candidates);

        // 7. D2H: top-k indices + candidate offsets
        std::vector<int> h_top_k_indices(actual_k);
        std::vector<size_t> candidate_offsets_cpu(num_candidates + 1);

        CUDA_CHECK(cudaMemcpyAsync(h_top_k_indices.data(), ws_.d_topk_indices,
                                   actual_k * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(candidate_offsets_cpu.data(), ws_.d_candidate_offsets,
                                   (num_candidates + 1) * sizeof(size_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Build output offsets on CPU
        std::vector<size_t> out_offsets(actual_k + 1, 0);
        for (size_t i = 0; i < actual_k; ++i) {
            size_t cand_idx = h_top_k_indices[i];
            out_offsets[i + 1] = out_offsets[i] +
                (candidate_offsets_cpu[cand_idx + 1] - candidate_offsets_cpu[cand_idx]);
        }
        size_t total_selected_tokens = out_offsets[actual_k];

        // H2D: upload selection metadata
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_selected_indices, h_top_k_indices.data(),
                                   actual_k * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_out_offsets, out_offsets.data(),
                                   (actual_k + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream));

        // 8. Naive extract kernel
        extract_one_bit_dists_baseline_kernel<<<actual_k, 256, 0, stream>>>(
            ws_.d_token_dists, ws_.d_candidate_offsets,
            ws_.d_selected_indices, ws_.d_out_one_bit_dists,
            ws_.d_out_offsets, total_tokens, actual_k
        );
        CUDA_CHECK(cudaGetLastError());

        // 9. D2H: extracted dists + doc IDs
        size_t copy_size = total_selected_tokens * Q_DOCLEN;
        CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_dists, ws_.d_out_one_bit_dists,
                                   copy_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_candidate_doc_ids, ws_.d_topk_doc_ids,
                                   num_candidates * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Build output doc IDs
        output_ids.resize(actual_k);
        for (size_t i = 0; i < actual_k; ++i) {
            output_ids[i] = ws_.h_pinned_candidate_doc_ids[h_top_k_indices[i]];
        }
    }

    // ======================== STAGE 3: CPU ========================
    // Identical to the optimized version — CPU ex-bits refinement.

    inline void rank_all_tokens_exbits_cpu(
        query_object* queries,
        std::vector<size_t>& input_ids,
        std::vector<float>& one_bit_dists,
        size_t k,
        std::vector<size_t>& output_ids
    ) {
        std::vector<size_t> candidate_doc_ptrs(input_ids.size() + 1);
        size_t total_tokens = 0;
        for (size_t i = 0; i < input_ids.size(); ++i) {
            total_tokens += doc_len(input_ids[i]);
            candidate_doc_ptrs[i + 1] = total_tokens;
        }
        std::priority_queue<std::pair<float, size_t>> max_heap;
#pragma omp parallel for
        for (size_t idx = 0; idx < input_ids.size(); ++idx) {
            size_t doc_id = input_ids[idx];
            float doc_score = 0.0F;
            for (size_t j = 0; j < Q_DOCLEN; ++j) {
                float max_token_score = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < doc_len(doc_id); ++i) {
                    size_t tid = doc_ptrs_[doc_id] + i;
                    float dist = distance_ex_bits(
                        queries + j,
                        &ex_code_[tid * PADDED_DIM * ex_bits / 8],
                        ex_bits,
                        ip_func_,
                        ws_.h_pinned_dists[(candidate_doc_ptrs[idx] + i) * Q_DOCLEN + j],
                        one_bit_factor_[tid],
                        ex_factor_[tid],
                        PADDED_DIM
                    );
                    max_token_score = std::max(max_token_score, dist);
                }
                doc_score += max_token_score;
            }
#pragma omp critical
            max_heap.emplace(doc_score, doc_id);
        }
        for (size_t i = 0; i < k && !max_heap.empty(); ++i) {
            output_ids.push_back(max_heap.top().second);
            max_heap.pop();
        }
    }

    ~gpu_mvr_index_baseline() {
        CUDA_CHECK(cudaFree(d_one_bit_code_));
        CUDA_CHECK(cudaFree(d_one_bit_factor_));
        CUDA_CHECK(cudaFree(d_doc_ids_));
        CUDA_CHECK(cudaFree(d_doc_ptrs_));
        CUDA_CHECK(cudaFree(d_inv_list_));
        CUDA_CHECK(cudaFree(d_cluster_pos_));

        CUDA_CHECK(cudaFree(ws_.d_queries));
        CUDA_CHECK(cudaFree(ws_.d_cb1_sumq));
        CUDA_CHECK(cudaFree(ws_.d_emb_ids));
        CUDA_CHECK(cudaFree(ws_.d_pair_offsets));
        CUDA_CHECK(cudaFree(ws_.d_emb_dists));
        CUDA_CHECK(cudaFree(ws_.d_pair_doc_ids));
        CUDA_CHECK(cudaFree(ws_.d_doc_query_max));
        CUDA_CHECK(cudaFree(ws_.d_stage1_doc_scores));
        CUDA_CHECK(cudaFree(ws_.d_topk_scores));
        CUDA_CHECK(cudaFree(ws_.d_topk_doc_ids));
        CUDA_CHECK(cudaFree(ws_.d_topk_indices));
        CUDA_CHECK(cudaFree(ws_.d_sorted_doc_ids));
        CUDA_CHECK(cudaFree(ws_.d_cub_temp_storage));
        CUDA_CHECK(cudaFree(ws_.d_token_ids));
        CUDA_CHECK(cudaFree(ws_.d_candidate_offsets));
        CUDA_CHECK(cudaFree(ws_.d_token_dists));
        CUDA_CHECK(cudaFree(ws_.d_doc_scores));
        CUDA_CHECK(cudaFree(ws_.d_selected_indices));
        CUDA_CHECK(cudaFree(ws_.d_out_offsets));
        CUDA_CHECK(cudaFree(ws_.d_out_one_bit_dists));
        CUDA_CHECK(cudaFree(ws_.d_cagra_dists));
        CUDA_CHECK(cudaFree(ws_.d_cagra_labels));

        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_queries));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_cb1_sumq));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_dists));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_candidate_doc_ids));

        CUDA_CHECK(cudaStreamDestroy(ws_.stream_compute));
        CUDA_CHECK(cudaStreamDestroy(ws_.stream_h2d));
        CUDA_CHECK(cudaStreamDestroy(ws_.stream_d2h));
        CUDA_CHECK(cudaEventDestroy(ws_.event_h2d_done));

        delete rotator_;
        delete ivf;
    }
};
