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

#include <oneapi/tbb.h>

// Prevent Eigen from adding __host__ __device__ annotations to its functions.
// Eigen is only used on the CPU side; without this, nvcc flags warning #20014-D
// for Eigen internals that call host-only STL functions.
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
    // padded_dim_ removed - using PADDED_DIM constant from gpu_config.cuh
    size_t num_docs;

    int max_doc_len = 0;       // for workspace sizing
    int max_cluster_size = 0;  // for workspace sizing

    // CPU-side data
    Rotator<float>* rotator_;
    IVF_PG* ivf;
    std::vector<char> one_bit_code_;     // CPU copy for compute-overlap path (1-bit IP on CPU)
    std::vector<char> ex_code_;          // CPU only - too large for GPU
    std::vector<float> one_bit_factor_;  // CPU copy for stage 3
    std::vector<float> ex_factor_;       // CPU only - stage 3 only
    std::vector<int> doc_ids_;           // CPU copy
    std::vector<int> doc_ptrs_;          // CPU copy for stage 3
    float (*ip_func_)(const float*, const uint8_t*, size_t);
    void (*unpack_func_)(const uint8_t*, float*, size_t);

    // GPU persistent data (copied once at construction)
    char*  d_one_bit_code_;     // [n * padded_dim_ / 8]
    float* d_one_bit_factor_;   // [n]
    int*   d_doc_ids_;          // [n]
    int*   d_doc_ptrs_;         // [num_docs + 1]
    int*   d_inv_list_;         // [inv_list.size()] - embedding IDs by cluster
    size_t* d_cluster_pos_;     // [n_clusters + 1] - cluster boundary offsets

    // Search parameters
    int nprobe = 128;
    int k_rank_cluster = 5000;    
    int k_rank_all_tokens = 300;

    // Pre-allocated GPU workspace (sized for worst-case per search)
    struct Workspace {
        // Query data (shared across stages)
        float* d_queries;       // [max_q_doclen * padded_dim]
        float* d_cb1_sumq;      // [max_q_doclen]

        // Stage 1 workspace
        size_t* d_emb_ids;      // [max_stage1_pairs]
        int*    d_pair_offsets;  // [max_q_doclen + 1]
        float*  d_emb_dists;    // [max_stage1_pairs]
        int*    d_pair_doc_ids; // [max_stage1_pairs]
        int*    d_pair_query_indices; // [max_stage1_pairs]

        // Stage 2 workspace
        size_t* d_token_ids;         // [max_stage2_tokens]
        size_t* d_candidate_offsets; // [max_stage2_candidates + 1]
        float*  d_token_dists;       // [max_stage2_tokens * max_q_doclen]
        float*  d_doc_scores;        // [max_stage2_candidates]

        // Extract workspace
        int*    d_selected_indices;  // [max_stage2_k]
        size_t* d_out_offsets;       // [max_stage2_k + 1]
        float*  d_out_one_bit_dists; // [max_stage2_k_tokens * max_q_doclen]

#ifdef GPU_MVR_USE_LUT
        float* d_lut;  // [Q_DOCLEN * LUT_ENTRIES_PER_QUERY] = 16384 floats = 64 KB
#endif

        // Stage 1 CAGRA batched search workspace
        float*        d_cagra_dists;     // [max_q_doclen * nprobe]
        faiss::idx_t* d_cagra_labels;    // [max_q_doclen * nprobe]

        // === NEW: GPU-only Stage 1 aggregation workspace ===
        int*    d_sorted_doc_ids;        // [max_stage1_pairs] for segmented reduction
        int*    d_sorted_query_indices;  // [max_stage1_pairs]
        float*  d_sorted_dists;          // [max_stage1_pairs]
        int*    d_unique_doc_ids;        // [estimated_num_docs]
        int*    d_doc_offsets;           // [estimated_num_docs + 1]
        float*  d_stage1_doc_scores;     // [estimated_num_docs]
        float*  d_doc_query_max;         // [estimated_num_docs * max_q_doclen] matrix
        int*    d_num_unique_docs;       // [1] output count
        int*    d_doc_touched;            // [estimated_num_docs] per-doc flag for sparse tracking
        void*   d_cub_temp_storage;      // CUB temporary buffer
        size_t  cub_temp_storage_bytes;  // Size of CUB buffer

        // === NEW: GPU top-k workspace ===
        float*  d_topk_scores;           // [max_stage2_candidates]
        int*    d_topk_doc_ids;          // [max_stage2_candidates]
        int*    d_topk_indices;          // [max_stage2_candidates]

        // Pinned host memory for fast H2D/D2H
        float*  h_pinned_queries;
        float*  h_pinned_cb1_sumq;
        float*  h_pinned_dists;
        float*  d_pinned_dists;              // device-mapped pointer for GPU direct write
        float*  h_pinned_batch_scores;       // [max_stage2_candidates] batch D2H

        size_t max_q_doclen;
        size_t max_stage1_pairs;
        size_t max_stage2_candidates;
        size_t max_stage2_tokens;
        size_t max_stage2_k;
        size_t max_stage2_k_tokens;
        size_t estimated_num_docs;
        int    max_embs_per_query_bound;  // precomputed upper bound: nprobe * max_cluster_size

        // === Persistent Stage 2+3 workspace (GPU_MVR_OVERLAP_STAGE23) ===
        // GPU computes binary IP + doc scores into HBM, then CPU sorts scores,
        // selects top-k, GPU extracts top-k dists, and CPU refines with overlap.
        size_t* d_pst_candidate_offsets;   // [max_stage2_candidates+1] device side
        size_t* d_pst_token_ids;           // [max_stage2_tokens] device side

        // Pinned host memory for D2H of doc scores
        float*        h_mapped_doc_scores;     // [max_stage2_candidates] pinned

        // CPU-side token offsets per candidate (pinned for async D2H)
        size_t* h_pinned_pst_candidate_offsets; // [max_stage2_candidates+1]
        int*    h_pinned_pst_candidate_doc_ids; // [max_stage2_candidates]
        size_t* h_pinned_pst_total_tokens;      // [1]

        // Pre-created Phase A profiling events (avoid per-call create/destroy)
        cudaEvent_t phase_a_start_event;
        cudaEvent_t phase_a_gather_done_event;
        cudaEvent_t phase_a_prefix_done_event;
        cudaEvent_t phase_a_token_ids_done_event;
        cudaEvent_t phase_a_d2h_done_event;

        // Pre-created events for overlap pipeline (avoid per-call create/destroy)
        cudaEvent_t pst_compute_done;     // after doc_score kernel
        cudaEvent_t pst_extract_done;     // after extract kernel
        static constexpr int PST_NUM_D2H_CHUNKS = 8;
        cudaEvent_t pst_d2h_chunk_done[PST_NUM_D2H_CHUNKS];  // per-chunk D2H events

        // === CUDA streams for async pipeline ===
        cudaStream_t stream_compute;
        cudaStream_t stream_h2d;
        cudaStream_t stream_d2h;

        // H2D completion event (reused across searches)
        cudaEvent_t event_h2d_done;

        // === NEW: Profiling events ===
#ifdef GPU_MVR_PROFILE
        cudaEvent_t event_start, event_end;
        cudaEvent_t event_stage1_start, event_stage1_end;
        cudaEvent_t event_stage2_start, event_stage2_end;

        // Stage 1 fine-grained kernel profiling events
        cudaEvent_t s1_cagra_start, s1_cagra_end;
        cudaEvent_t s1_expansion_start, s1_expansion_end;
        cudaEvent_t s1_binary_ip_start, s1_binary_ip_end;
        cudaEvent_t s1_memset_start, s1_memset_end;
        cudaEvent_t s1_atomic_agg_start, s1_atomic_agg_end;
        cudaEvent_t s1_sum_scores_start, s1_sum_scores_end;
        cudaEvent_t s1_topk_sort_start, s1_topk_sort_end;
        cudaEvent_t s1_d2d_start, s1_d2d_end;

        // Stage 2 fine-grained profiling events (per-batch, reused)
        cudaEvent_t s2_gather_start, s2_gather_end;
        cudaEvent_t s2_prefix_start, s2_prefix_end;
        cudaEvent_t s2_tokenids_start, s2_tokenids_end;
        cudaEvent_t s2_binaryip_start, s2_binaryip_end;
        cudaEvent_t s2_docscore_start, s2_docscore_end;
        cudaEvent_t s2_d2h_start, s2_d2h_end;
        cudaEvent_t s2_extract_start, s2_extract_end;

        // Stage 2 accumulated timings (ms)
        float s2_gather_ms   = 0;
        float s2_prefix_ms   = 0;
        float s2_tokenids_ms = 0;
        float s2_binaryip_ms = 0;
        float s2_docscore_ms = 0;
        float s2_d2h_ms      = 0;
        float s2_topk_cpu_us = 0;  // CPU-side top-k (microseconds)
        float s2_extract_ms  = 0;
        int   s2_num_batches = 0;

        // Persistent Stage 2+3 profiling events
        cudaEvent_t s23_pst_kernel_start, s23_pst_kernel_end;
        // Accumulated timings for persistent path
        float s23_pst_kernel_ms   = 0;  // Phase B total GPU kernel wall time
        float s23_pst_bip_ms      = 0;  // Phase B binary_ip kernel total time
        float s23_pst_docscore_ms = 0;  // Phase B doc_score_kernel total time
        float s23_cpu_total_us   = 0;  // total CPU Stage 3 time (microseconds)
        float s23_heap_total_us  = 0;  // total heap update time (microseconds)
        int   s23_total_refined  = 0;  // total docs refined in Stage 3

        // === Transfer profiling ===
        static constexpr int MAX_XFER_RECORDS = 48;
        struct XferRecord {
            cudaEvent_t start, end;
            size_t bytes;
            bool is_h2d;  // true=H2D, false=D2H
        };
        XferRecord xfer_records[MAX_XFER_RECORDS];
        int xfer_count = 0;
#endif
        // Phase C reusable CPU buffers (pre-allocated to avoid per-search heap allocation)
        std::vector<int>                   running_indices;
        std::vector<int>                   h_sel_indices;
        std::vector<size_t>                h_out_offsets;
        std::vector<float>                 ip_ex_buf;
        std::vector<std::pair<float, int>> refined_scores;
    } ws_;

    // Construct from file
    gpu_mvr_index(const std::string& filename, const std::vector<int>& doc_lens) {
        std::ifstream inf(filename, std::ios::binary);
        inf.read((char*)&n, sizeof(size_t));
        inf.read((char*)&d, sizeof(size_t));
        inf.read((char*)&n_clusters, sizeof(size_t));
        inf.read((char*)&ex_bits, sizeof(size_t));

        // Read padded_dim from file and validate it matches compiled constant
        size_t file_padded_dim;
        inf.read((char*)&file_padded_dim, sizeof(size_t));

        if (file_padded_dim != PADDED_DIM) {
            inf.close();
            throw std::runtime_error(
                "Index file padded_dim=" + std::to_string(file_padded_dim) +
                " does not match compiled PADDED_DIM=" + std::to_string(PADDED_DIM) +
                ". Please recompile with matching PADDED_DIM in gpu_config.cuh"
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

        // Upload inverted list structures to GPU
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

        // Allocate workspace
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
            for (size_t j = 0; j < doc_lens[i]; ++j) {
                doc_ids_[doc_ptrs_[i] + j] = i;
            }
        }
    }

    inline void allocate_workspace() {
        // Estimate worst-case sizes
        // Stage 1: nprobe clusters * max_cluster_size * Q_DOCLEN pairs
        ws_.max_q_doclen = Q_DOCLEN;
        ws_.max_stage1_pairs = nprobe * max_cluster_size * Q_DOCLEN;
        ws_.max_stage2_candidates = k_rank_cluster;  // k_rank_cluster
        // Estimate avg ~100 tokens per doc
        ws_.max_stage2_tokens = ws_.max_stage2_candidates * max_doc_len;
        ws_.max_stage2_k = k_rank_all_tokens;  // k_rank_all_tokens
        ws_.max_stage2_k_tokens = ws_.max_stage2_k * max_doc_len;
        ws_.estimated_num_docs = (size_t)num_docs;

        // Query workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_cb1_sumq, Q_DOCLEN * sizeof(float)));

        // Stage 1 workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_emb_ids, ws_.max_stage1_pairs * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_pair_offsets, (Q_DOCLEN + 1) * sizeof(int)));

        // OPTIMIZATION: Allocate emb_dists for [query][embedding] layout
        // Worst case: Q_DOCLEN * (max_stage1_pairs / Q_DOCLEN) = max_stage1_pairs
        // But to handle uneven distribution, allocate Q_DOCLEN * max_stage1_pairs/Q_DOCLEN with padding
        // For safety, we allocate the full max_stage1_pairs which is always sufficient
        // since sum(embs_per_query[i]) <= max_stage1_pairs
        CUDA_CHECK(cudaMalloc(&ws_.d_emb_dists, ws_.max_stage1_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_pair_doc_ids, ws_.max_stage1_pairs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_pair_query_indices, ws_.max_stage1_pairs * sizeof(int)));

        // === NEW: Stage 1 GPU aggregation workspace ===
        CUDA_CHECK(cudaMalloc(&ws_.d_sorted_doc_ids, ws_.max_stage1_pairs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_sorted_query_indices, ws_.max_stage1_pairs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_sorted_dists, ws_.max_stage1_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_unique_doc_ids, ws_.estimated_num_docs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_doc_offsets, (ws_.estimated_num_docs + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_stage1_doc_scores, ws_.estimated_num_docs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_doc_query_max, ws_.estimated_num_docs * Q_DOCLEN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_num_unique_docs, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_doc_touched, ws_.estimated_num_docs * sizeof(int)));
        
        // Query CUB temporary storage size for ALL operations (dry runs)
        ws_.d_cub_temp_storage = nullptr;
        ws_.cub_temp_storage_bytes = 0;

        size_t temp_bytes = 0;

        // Stage 1: SortPairs for embedding pairs
        temp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, temp_bytes,
            ws_.d_sorted_doc_ids, ws_.d_sorted_doc_ids,
            ws_.d_sorted_dists, ws_.d_sorted_dists,
            ws_.max_stage1_pairs
        );
        ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);

        // Stage 1: SortPairsDescending for doc scores (sorts num_docs keys)
        temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_bytes,
            ws_.d_stage1_doc_scores, ws_.d_topk_scores,
            ws_.d_topk_indices, ws_.d_sorted_doc_ids,
            (int)num_docs, 0, 32
        );
        ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);

#ifndef GPU_MVR_OVERLAP_STAGE23
        // Stage 2: SortPairsDescending for candidate scores (non-overlapping path only)
        temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_bytes,
            ws_.d_doc_scores, ws_.d_topk_scores,
            ws_.d_selected_indices, ws_.d_topk_indices,
            (int)ws_.max_stage2_candidates, 0, 32
        );
        ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);
#endif

        // Stage 1: InclusiveScan for pair offsets (int -> int)
        temp_bytes = 0;
        cub::DeviceScan::InclusiveScan(
            nullptr, temp_bytes,
            ws_.d_pair_offsets + 1, ws_.d_pair_offsets + 1,
            thrust::plus<size_t>(), (int)Q_DOCLEN
        );
        ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);

        // Stage 2: InclusiveScan for candidate offsets (int -> size_t with transform)
        {
            thrust::device_ptr<int> len_ptr(ws_.d_pair_doc_ids);
            auto xform_iter = thrust::make_transform_iterator(len_ptr, cast_int_size_t());
            temp_bytes = 0;
            cub::DeviceScan::InclusiveScan(
                nullptr, temp_bytes,
                xform_iter, ws_.d_candidate_offsets + 1,
                thrust::plus<size_t>(), (int)ws_.max_stage2_candidates
            );
            ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);
        }

        CUDA_CHECK(cudaMalloc(&ws_.d_cub_temp_storage, ws_.cub_temp_storage_bytes));

        // === NEW: GPU top-k workspace (must fit num_docs for Stage 1 sort) ===
        size_t topk_buf_size = std::max((size_t)num_docs, ws_.max_stage2_candidates);
        CUDA_CHECK(cudaMalloc(&ws_.d_topk_scores, topk_buf_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_topk_doc_ids, topk_buf_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_topk_indices, topk_buf_size * sizeof(int)));

#ifndef GPU_MVR_OVERLAP_STAGE23
        // Stage 2 workspace (non-overlapping path: processes all candidates at once)
        CUDA_CHECK(cudaMalloc(&ws_.d_token_ids, ws_.max_stage2_tokens * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_candidate_offsets, (ws_.max_stage2_candidates + 1) * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_token_dists, ws_.max_stage2_tokens * Q_DOCLEN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_doc_scores, ws_.max_stage2_candidates * sizeof(float)));
#endif

        // Extract workspace (used by both paths)
        CUDA_CHECK(cudaMalloc(&ws_.d_selected_indices, ws_.max_stage2_k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_out_offsets, (ws_.max_stage2_k + 1) * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_out_one_bit_dists, ws_.max_stage2_k_tokens * Q_DOCLEN * sizeof(float)));

#ifdef GPU_MVR_USE_LUT
        // LUT workspace for binary IP
        CUDA_CHECK(cudaMalloc(&ws_.d_lut, LUT_TOTAL_FLOATS * sizeof(float)));
        // Stage 2 LUT kernel needs extended shared memory (64 KB + 128 B)
        size_t stage2_lut_smem = STAGE2_LUT_SMEM_FLOATS * sizeof(float) + STAGE2_LUT_TILE_Q * sizeof(float);
        CUDA_CHECK(cudaFuncSetAttribute(
            stage2_binary_ip_lut_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            stage2_lut_smem));
#endif

        // CAGRA batched search workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_cagra_dists, Q_DOCLEN * nprobe * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_cagra_labels, Q_DOCLEN * nprobe * sizeof(faiss::idx_t)));

        // Pinned host memory
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_cb1_sumq, Q_DOCLEN * sizeof(float)));
        CUDA_CHECK(cudaHostAlloc(&ws_.h_pinned_dists, ws_.max_stage2_k_tokens * Q_DOCLEN * sizeof(float), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&ws_.d_pinned_dists, ws_.h_pinned_dists, 0));
#ifndef GPU_MVR_OVERLAP_STAGE23
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_batch_scores, ws_.max_stage2_candidates * sizeof(float)));
#endif

#ifdef GPU_MVR_OVERLAP_STAGE23
        // === Persistent Stage 2+3 workspace ===
        CUDA_CHECK(cudaMalloc(&ws_.d_pst_candidate_offsets,
                              (ws_.max_stage2_candidates + 1) * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_pst_token_ids,
                              ws_.max_stage2_tokens * sizeof(size_t)));

        // Device HBM buffer for token distances and doc scores
        CUDA_CHECK(cudaMalloc(&ws_.d_token_dists,
                              ws_.max_stage2_tokens * (size_t)Q_DOCLEN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_doc_scores, ws_.max_stage2_candidates * sizeof(float)));

        // Pinned host memory for D2H of doc scores
        CUDA_CHECK(cudaMallocHost(&ws_.h_mapped_doc_scores,
                                  ws_.max_stage2_candidates * sizeof(float)));

        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_pst_candidate_offsets,
                                  (ws_.max_stage2_candidates + 1) * sizeof(size_t)));
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_pst_candidate_doc_ids,
                                  ws_.max_stage2_candidates * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_pst_total_tokens, sizeof(size_t)));

        // Pre-create Phase A profiling events
        CUDA_CHECK(cudaEventCreate(&ws_.phase_a_start_event));
        CUDA_CHECK(cudaEventCreate(&ws_.phase_a_gather_done_event));
        CUDA_CHECK(cudaEventCreate(&ws_.phase_a_prefix_done_event));
        CUDA_CHECK(cudaEventCreate(&ws_.phase_a_token_ids_done_event));
        CUDA_CHECK(cudaEventCreate(&ws_.phase_a_d2h_done_event));

        // Pre-create events for overlap pipeline
        CUDA_CHECK(cudaEventCreateWithFlags(&ws_.pst_compute_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ws_.pst_extract_done, cudaEventDisableTiming));
        for (int i = 0; i < Workspace::PST_NUM_D2H_CHUNKS; i++)
            CUDA_CHECK(cudaEventCreateWithFlags(&ws_.pst_d2h_chunk_done[i], cudaEventDisableTiming));
#endif

        // Initialize persistent matrices (cleaned per-search with sparse cleanup)
        ws_.max_embs_per_query_bound = nprobe * max_cluster_size;
        CUDA_CHECK(cudaMemset(ws_.d_doc_query_max, 0, ws_.estimated_num_docs * Q_DOCLEN * sizeof(float)));
        CUDA_CHECK(cudaMemset(ws_.d_doc_touched, 0, ws_.estimated_num_docs * sizeof(int)));
        CUDA_CHECK(cudaMemset(ws_.d_num_unique_docs, 0, sizeof(int)));

        // === Create CUDA streams ===
        CUDA_CHECK(cudaStreamCreate(&ws_.stream_compute));
        CUDA_CHECK(cudaStreamCreate(&ws_.stream_h2d));
        CUDA_CHECK(cudaStreamCreate(&ws_.stream_d2h));
        CUDA_CHECK(cudaEventCreate(&ws_.event_h2d_done));

        // === NEW: Create profiling events ===
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventCreate(&ws_.event_start));
        CUDA_CHECK(cudaEventCreate(&ws_.event_end));
        CUDA_CHECK(cudaEventCreate(&ws_.event_stage1_start));
        CUDA_CHECK(cudaEventCreate(&ws_.event_stage1_end));
        CUDA_CHECK(cudaEventCreate(&ws_.event_stage2_start));
        CUDA_CHECK(cudaEventCreate(&ws_.event_stage2_end));

        // Stage 1 fine-grained kernel events
        CUDA_CHECK(cudaEventCreate(&ws_.s1_cagra_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_cagra_end));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_expansion_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_expansion_end));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_binary_ip_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_binary_ip_end));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_memset_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_memset_end));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_atomic_agg_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_atomic_agg_end));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_sum_scores_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_sum_scores_end));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_topk_sort_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_topk_sort_end));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_d2d_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s1_d2d_end));

        CUDA_CHECK(cudaEventCreate(&ws_.s2_gather_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s2_gather_end));
        CUDA_CHECK(cudaEventCreate(&ws_.s2_prefix_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s2_prefix_end));
        CUDA_CHECK(cudaEventCreate(&ws_.s2_tokenids_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s2_tokenids_end));
        CUDA_CHECK(cudaEventCreate(&ws_.s2_binaryip_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s2_binaryip_end));
        CUDA_CHECK(cudaEventCreate(&ws_.s2_docscore_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s2_docscore_end));
        CUDA_CHECK(cudaEventCreate(&ws_.s2_d2h_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s2_d2h_end));
        CUDA_CHECK(cudaEventCreate(&ws_.s2_extract_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s2_extract_end));

        // Persistent Stage 2+3 profiling events
        CUDA_CHECK(cudaEventCreate(&ws_.s23_pst_kernel_start));
        CUDA_CHECK(cudaEventCreate(&ws_.s23_pst_kernel_end));

        // Transfer profiling events
        for (int i = 0; i < Workspace::MAX_XFER_RECORDS; i++) {
            CUDA_CHECK(cudaEventCreate(&ws_.xfer_records[i].start));
            CUDA_CHECK(cudaEventCreate(&ws_.xfer_records[i].end));
        }
        ws_.xfer_count = 0;
#endif
        // Pre-allocate Phase C reusable CPU buffers
        ws_.running_indices.resize(ws_.max_stage2_candidates);
        ws_.h_sel_indices.resize(ws_.max_stage2_k);
        ws_.h_out_offsets.resize(ws_.max_stage2_k + 1);
        ws_.ip_ex_buf.resize(ws_.max_stage2_k * (size_t)max_doc_len * Q_DOCLEN);
        ws_.refined_scores.resize(ws_.max_stage2_k);
    }

    inline size_t doc_len(size_t doc_id) const {
        return doc_ptrs_[doc_id + 1] - doc_ptrs_[doc_id];
    }

    // ======================== SEARCH PIPELINE ========================

    inline std::vector<size_t> search(const float* queries, size_t k) {
#ifdef GPU_MVR_PROFILE
        auto search_start = std::chrono::high_resolution_clock::now();
        ws_.xfer_count = 0;  // reset transfer profiling counters
#endif

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_start, ws_.stream_compute));
#endif

        // Step 1: Rotate queries on CPU into pinned memory (async-ready)
        for (size_t i = 0; i < Q_DOCLEN; ++i) {
            rotator_->rotate(&queries[i * d], &ws_.h_pinned_queries[i * PADDED_DIM]);
        }

        // Build query objects on CPU
        std::vector<query_object> query_objs(Q_DOCLEN);
        for (size_t i = 0; i < Q_DOCLEN; ++i) {
            query_objs[i] = query_object(&ws_.h_pinned_queries[i * PADDED_DIM], PADDED_DIM, ex_bits);
            ws_.h_pinned_cb1_sumq[i] = query_objs[i].cb1_sumq;
        }

        // Upload from pinned memory using async stream
        XFER_RECORD_BEGIN(ws_.stream_h2d);
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_queries, ws_.h_pinned_queries,
                                   Q_DOCLEN * PADDED_DIM * sizeof(float),
                                   cudaMemcpyHostToDevice, ws_.stream_h2d));
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_cb1_sumq, ws_.h_pinned_cb1_sumq,
                                   Q_DOCLEN * sizeof(float),
                                   cudaMemcpyHostToDevice, ws_.stream_h2d));
        XFER_RECORD_END(ws_.stream_h2d, Q_DOCLEN * PADDED_DIM * sizeof(float) + Q_DOCLEN * sizeof(float), true);

        // Wait for H2D transfer before compute
        CUDA_CHECK(cudaEventRecord(ws_.event_h2d_done, ws_.stream_h2d));
        CUDA_CHECK(cudaStreamWaitEvent(ws_.stream_compute, ws_.event_h2d_done));

#ifdef GPU_MVR_USE_LUT
        // Precompute LUT for all queries (must happen after H2D of d_queries)
        precompute_lut_kernel<<<Q_DOCLEN, 256, 0, ws_.stream_compute>>>(
            ws_.d_queries, ws_.d_lut);
        CUDA_CHECK(cudaGetLastError());
#endif

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_stage1_start, ws_.stream_compute));
#endif

        // Stage 1: GPU-only aggregation and top-k
        int actual_k_stage1 = 0;
        rank_cluster_dists_gpu(query_objs.data(), nprobe, k_rank_cluster,
                              actual_k_stage1, ws_.stream_compute);

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_stage1_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventRecord(ws_.event_stage2_start, ws_.stream_compute));
#endif

        std::vector<size_t> result;

#ifdef GPU_MVR_OVERLAP_STAGE23
        rank_stage23_persistent(actual_k_stage1, k, k_rank_all_tokens, query_objs.data(), result);

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_stage2_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventRecord(ws_.event_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventSynchronize(ws_.event_end));

        float total_time, stage1_time;
        float s1_cagra, s1_expansion, s1_binary_ip, s1_memset, s1_atomic_agg, s1_sum_scores, s1_topk_sort, s1_d2d;
        CUDA_CHECK(cudaEventElapsedTime(&total_time, ws_.event_start, ws_.event_end));
        CUDA_CHECK(cudaEventElapsedTime(&stage1_time, ws_.event_stage1_start, ws_.event_stage1_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_cagra, ws_.s1_cagra_start, ws_.s1_cagra_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_expansion, ws_.s1_expansion_start, ws_.s1_expansion_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_binary_ip, ws_.s1_binary_ip_start, ws_.s1_binary_ip_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_memset, ws_.s1_memset_start, ws_.s1_memset_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_atomic_agg, ws_.s1_atomic_agg_start, ws_.s1_atomic_agg_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_sum_scores, ws_.s1_sum_scores_start, ws_.s1_sum_scores_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_topk_sort, ws_.s1_topk_sort_start, ws_.s1_topk_sort_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_d2d, ws_.s1_d2d_start, ws_.s1_d2d_end));

        std::cout << "[PROFILE] Mode: Persistent Stage 2+3 (streaming top-k + system fence)\n";
        std::cout << "[PROFILE] Stage 1 time: " << stage1_time << " ms\n";
        std::cout << "[PROFILE]   1. CAGRA search            : " << s1_cagra << " ms\n";
        std::cout << "[PROFILE]   2. GPU IVF expansion       : " << s1_expansion << " ms\n";
        std::cout << "[PROFILE]   3. Binary IP kernel        : " << s1_binary_ip << " ms\n";
        std::cout << "[PROFILE]   4. Aggregation + tracking  : " << s1_atomic_agg << " ms\n";
        std::cout << "[PROFILE]   5. Sum doc scores (sparse) : " << s1_sum_scores << " ms\n";
        std::cout << "[PROFILE]   6. Top-k sort (sparse)     : " << s1_topk_sort << " ms\n";
        std::cout << "[PROFILE]   7. D2D copy top-k doc IDs  : " << s1_d2d << " ms\n";
        std::cout << "[PROFILE]   8. Memset (overlapped 1-3) : " << s1_memset << " ms (not in critical path)\n";
        float s1_sum = s1_cagra + s1_expansion + s1_binary_ip + s1_atomic_agg + s1_sum_scores + s1_topk_sort + s1_d2d;
        std::cout << "[PROFILE]   Sum accounted              : " << s1_sum << " ms\n";
        std::cout << "[PROFILE] Phase B binary_ip total     : " << ws_.s23_pst_bip_ms << " ms\n";
        std::cout << "[PROFILE] Phase B doc_score total     : " << ws_.s23_pst_docscore_ms << " ms\n";
        std::cout << "[PROFILE] Phase B total kernel time   : " << ws_.s23_pst_kernel_ms << " ms\n";
        std::cout << "[PROFILE] Total search time           : " << total_time << " ms\n";

        // === Transfer profiling summary ===
        {
            float total_h2d_ms = 0, total_d2h_ms = 0;
            size_t total_h2d_bytes = 0, total_d2h_bytes = 0;
            for (int i = 0; i < ws_.xfer_count; i++) {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ws_.xfer_records[i].start, ws_.xfer_records[i].end));
                if (ws_.xfer_records[i].is_h2d) {
                    total_h2d_ms += ms;
                    total_h2d_bytes += ws_.xfer_records[i].bytes;
                } else {
                    total_d2h_ms += ms;
                    total_d2h_bytes += ws_.xfer_records[i].bytes;
                }
            }
            std::cout << "[PROFILE] Data transfer summary (" << ws_.xfer_count << " transfers):\n";
            std::cout << "[PROFILE]   H2D: " << total_h2d_ms << " ms, "
                      << total_h2d_bytes << " bytes (" << (total_h2d_bytes / 1024.0) << " KB)\n";
            std::cout << "[PROFILE]   D2H: " << total_d2h_ms << " ms, "
                      << total_d2h_bytes << " bytes (" << (total_d2h_bytes / 1024.0) << " KB)\n";
            std::cout << "[PROFILE]   Total transfer: " << (total_h2d_ms + total_d2h_ms) << " ms, "
                      << (total_h2d_bytes + total_d2h_bytes) << " bytes ("
                      << ((total_h2d_bytes + total_d2h_bytes) / 1024.0) << " KB)\n";
        }
#endif

#else  // !GPU_MVR_OVERLAP_STAGE23
        // === Non-overlapping: Stage 2 fully, then Stage 3 fully ===
        std::vector<size_t> stage2_doc_ids;
        std::vector<float>  stage2_one_bit_dists;

        rank_all_tokens_1bit_gpu(actual_k_stage1, k_rank_all_tokens,
                                 stage2_doc_ids, stage2_one_bit_dists,
                                 ws_.stream_compute);

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_stage2_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventSynchronize(ws_.event_stage2_end));
#endif

        // Stage 3: CPU ex-bits refinement (sequential, no overlap)
#ifdef GPU_MVR_PROFILE
        auto stage3_wall_start = std::chrono::high_resolution_clock::now();
#endif

        rank_all_tokens_exbits_cpu(query_objs.data(),
                                   stage2_doc_ids, stage2_one_bit_dists,
                                   k, result);

#ifdef GPU_MVR_PROFILE
        auto stage3_wall_end = std::chrono::high_resolution_clock::now();
        float stage3_time_ms = std::chrono::duration<float, std::milli>(
            stage3_wall_end - stage3_wall_start).count();

        CUDA_CHECK(cudaEventRecord(ws_.event_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventSynchronize(ws_.event_end));

        float total_time, stage1_time, stage2_time;
        float s1_cagra, s1_expansion, s1_binary_ip, s1_memset, s1_atomic_agg, s1_sum_scores, s1_topk_sort, s1_d2d;
        CUDA_CHECK(cudaEventElapsedTime(&total_time, ws_.event_start, ws_.event_end));
        CUDA_CHECK(cudaEventElapsedTime(&stage1_time, ws_.event_stage1_start, ws_.event_stage1_end));
        CUDA_CHECK(cudaEventElapsedTime(&stage2_time, ws_.event_stage2_start, ws_.event_stage2_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_cagra, ws_.s1_cagra_start, ws_.s1_cagra_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_expansion, ws_.s1_expansion_start, ws_.s1_expansion_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_binary_ip, ws_.s1_binary_ip_start, ws_.s1_binary_ip_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_memset, ws_.s1_memset_start, ws_.s1_memset_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_atomic_agg, ws_.s1_atomic_agg_start, ws_.s1_atomic_agg_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_sum_scores, ws_.s1_sum_scores_start, ws_.s1_sum_scores_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_topk_sort, ws_.s1_topk_sort_start, ws_.s1_topk_sort_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_d2d, ws_.s1_d2d_start, ws_.s1_d2d_end));

        std::cout << "[PROFILE] Mode: Non-overlapping Stage 2 then 3\n";
        std::cout << "[PROFILE] Total GPU time: " << total_time << " ms\n";
        std::cout << "[PROFILE] Stage 1 time: " << stage1_time << " ms\n";
        std::cout << "[PROFILE]   1. CAGRA search            : " << s1_cagra << " ms\n";
        std::cout << "[PROFILE]   2. GPU IVF expansion       : " << s1_expansion << " ms\n";
        std::cout << "[PROFILE]   3. Binary IP kernel        : " << s1_binary_ip << " ms\n";
        std::cout << "[PROFILE]   4. Aggregation + tracking  : " << s1_atomic_agg << " ms\n";
        std::cout << "[PROFILE]   5. Sum doc scores (sparse) : " << s1_sum_scores << " ms\n";
        std::cout << "[PROFILE]   6. Top-k sort (sparse)     : " << s1_topk_sort << " ms\n";
        std::cout << "[PROFILE]   7. D2D copy top-k doc IDs  : " << s1_d2d << " ms\n";
        std::cout << "[PROFILE]   8. Memset (overlapped 1-3) : " << s1_memset << " ms (not in critical path)\n";
        float s1_sum = s1_cagra + s1_expansion + s1_binary_ip + s1_atomic_agg + s1_sum_scores + s1_topk_sort + s1_d2d;
        std::cout << "[PROFILE]   Sum accounted              : " << s1_sum << " ms\n";
        std::cout << "[PROFILE] Stage 2 time: " << stage2_time << " ms\n";
        std::cout << "[PROFILE] Stage 3 time: " << stage3_time_ms << " ms"
                  << " (" << stage2_doc_ids.size() << " docs)\n";

        // === Transfer profiling summary ===
        {
            float total_h2d_ms = 0, total_d2h_ms = 0;
            size_t total_h2d_bytes = 0, total_d2h_bytes = 0;
            for (int i = 0; i < ws_.xfer_count; i++) {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ws_.xfer_records[i].start, ws_.xfer_records[i].end));
                if (ws_.xfer_records[i].is_h2d) {
                    total_h2d_ms += ms;
                    total_h2d_bytes += ws_.xfer_records[i].bytes;
                } else {
                    total_d2h_ms += ms;
                    total_d2h_bytes += ws_.xfer_records[i].bytes;
                }
            }
            std::cout << "[PROFILE] Data transfer summary (" << ws_.xfer_count << " transfers):\n";
            std::cout << "[PROFILE]   H2D: " << total_h2d_ms << " ms, "
                      << total_h2d_bytes << " bytes (" << (total_h2d_bytes / 1024.0) << " KB)\n";
            std::cout << "[PROFILE]   D2H: " << total_d2h_ms << " ms, "
                      << total_d2h_bytes << " bytes (" << (total_d2h_bytes / 1024.0) << " KB)\n";
            std::cout << "[PROFILE]   Total transfer: " << (total_h2d_ms + total_d2h_ms) << " ms, "
                      << (total_h2d_bytes + total_d2h_bytes) << " bytes ("
                      << ((total_h2d_bytes + total_d2h_bytes) / 1024.0) << " KB)\n";
        }
#endif

#endif  // GPU_MVR_OVERLAP_STAGE23


        return result;
    }

    // ======================== STAGE 1: GPU ========================
    // GPU-only approach: eliminates CPU aggregation round-trip

    inline void rank_cluster_dists_gpu(
        query_object* h_query_objs,
        size_t nprobe, size_t k,
        int& actual_k_out,
        cudaStream_t stream = 0
    ) {
        // Launch memset on d2h stream (overlaps with CAGRA + expansion + binary IP)
        size_t matrix_size = (size_t)num_docs * Q_DOCLEN;
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_memset_start, ws_.stream_d2h));
#endif
        CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_query_max, 0, matrix_size * sizeof(float), ws_.stream_d2h));
        CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_touched, 0, ws_.estimated_num_docs * sizeof(int), ws_.stream_d2h));
        CUDA_CHECK(cudaMemsetAsync(ws_.d_num_unique_docs, 0, sizeof(int), ws_.stream_d2h));
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_memset_end, ws_.stream_d2h));
#endif
        CUDA_CHECK(cudaEventRecord(ws_.event_h2d_done, ws_.stream_d2h));  // reuse for memset sync

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_cagra_start, stream));
#endif

        // 1. Batched CAGRA search: one call for all Q_DOCLEN queries on GPU
        //    ws_.d_queries is already on GPU from the caller.
        ivf->search_batch_gpu(ws_.d_queries, Q_DOCLEN, nprobe,
                              ws_.d_cagra_dists, ws_.d_cagra_labels);

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_cagra_end, stream));
        CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_start, stream));
#endif

        // GPU-based cluster expansion (replaces CPU expansion and transfers)
        // Step 1: Compute per-query expansion sizes on GPU
        compute_query_expansion_sizes_kernel<<<(Q_DOCLEN + 255) / 256, 256, 0, stream>>>(
            ws_.d_cagra_labels,           // Already on GPU from CAGRA search
            d_cluster_pos_,
            ws_.d_pair_offsets + 1,       // Write sizes to offset[1..Q_DOCLEN]
            nprobe,
            ivf->n_clusters,
            Q_DOCLEN
        );

        // Step 2: Set first offset to 0
        // int zero = 0;
        // CUDA_CHECK(cudaMemcpyAsync(ws_.d_pair_offsets, &zero, sizeof(int),
        //                            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(ws_.d_pair_offsets, 0, sizeof(int), stream));

        // Step 3: Compute prefix sum for offsets
        {
            size_t scan_temp = ws_.cub_temp_storage_bytes;
            cub::DeviceScan::InclusiveScan(
                ws_.d_cub_temp_storage, scan_temp,
                ws_.d_pair_offsets + 1, ws_.d_pair_offsets + 1,
                thrust::plus<int>(), (int)Q_DOCLEN, stream);
        }

        // Step 4: Get total pairs count
        int total_pairs;
        XFER_RECORD_BEGIN(stream);
        CUDA_CHECK(cudaMemcpyAsync(&total_pairs, ws_.d_pair_offsets + Q_DOCLEN,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream));
        XFER_RECORD_END(stream, sizeof(int), false);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (total_pairs == 0) {
            actual_k_out = 0;
            return;
        }

        // Step 5: Launch expansion kernel
        expand_cluster_ids_kernel<<<Q_DOCLEN, 256, 0, stream>>>(
            ws_.d_cagra_labels,
            d_inv_list_,
            d_cluster_pos_,
            ws_.d_pair_offsets,
            ws_.d_emb_ids,               // Direct write to GPU workspace
            nprobe,
            ivf->n_clusters,
            Q_DOCLEN
        );

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_end, stream));
#endif

        // Note: ws_.d_emb_ids and ws_.d_pair_offsets are now ready for Stage 1 kernel

        // 2. Compute 1-bit distances using optimized shared-memory kernel
        // Use precomputed upper bound for max_embs_per_query
        // (eliminates thrust::device_vector alloc + adjacent_difference + reduce per search)
        int max_embs_per_query = ws_.max_embs_per_query_bound;

        int threads_per_block = 256;
        int blocks_x = (max_embs_per_query + threads_per_block - 1) / threads_per_block;

        dim3 grid(blocks_x, Q_DOCLEN);
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_start, stream));
#endif
#ifdef GPU_MVR_USE_LUT
        stage1_binary_ip_lut_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_lut, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
            ws_.d_emb_ids, ws_.d_pair_offsets, ws_.d_emb_dists,
            max_embs_per_query
        );
#else
        stage1_binary_ip_kernel_v2<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_queries, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
            ws_.d_emb_ids, ws_.d_pair_offsets, ws_.d_emb_dists,
            max_embs_per_query
        );
#endif
        CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_end, stream));
#endif

        // === Sparse aggregation: only process documents actually hit ===
        // Wait for overlapped memset (on stream_d2h) to complete before aggregation
        CUDA_CHECK(cudaStreamWaitEvent(stream, ws_.event_h2d_done));

        // 4. Atomic aggregation with touched-doc tracking
        int thread_count = 256;
        int block_count = (total_pairs + thread_count - 1) / thread_count;

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_start, stream));
#endif
        aggregate_stage1_tracked_kernel<<<block_count, thread_count, 0, stream>>>(
            ws_.d_emb_ids, ws_.d_emb_dists, ws_.d_pair_offsets, d_doc_ids_,
            ws_.d_doc_query_max,
            ws_.d_doc_touched, ws_.d_unique_doc_ids, ws_.d_num_unique_docs,
            num_docs, total_pairs, max_embs_per_query
        );
        CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_end, stream));
#endif

        // 5. Get number of touched docs (single int D2H)
        int h_num_touched = 0;
        XFER_RECORD_BEGIN(stream);
        CUDA_CHECK(cudaMemcpyAsync(&h_num_touched, ws_.d_num_unique_docs,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream));
        XFER_RECORD_END(stream, sizeof(int), false);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (h_num_touched == 0) {
            actual_k_out = 0;
            return;
        }
        actual_k_out = std::min((int)k, h_num_touched);

        // 6. Sum across queries only for touched docs (sparse)
        int sparse_blocks = (h_num_touched + thread_count - 1) / thread_count;
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_sum_scores_start, stream));
#endif
        sum_doc_scores_sparse_kernel<<<sparse_blocks, thread_count, 0, stream>>>(
            ws_.d_doc_query_max,
            ws_.d_unique_doc_ids,
            ws_.d_stage1_doc_scores,   // compact scores [num_touched]
            ws_.d_topk_doc_ids,        // compact doc IDs [num_touched]
            h_num_touched, num_docs
        );
        CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_sum_scores_end, stream));
#endif

        // 7. Top-k: sort only touched docs by score descending
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_topk_sort_start, stream));
#endif
        cub::DeviceRadixSort::SortPairsDescending(
            ws_.d_cub_temp_storage, ws_.cub_temp_storage_bytes,
            ws_.d_stage1_doc_scores, ws_.d_topk_scores,
            ws_.d_topk_doc_ids, ws_.d_sorted_doc_ids,
            h_num_touched, 0, 32, stream
        );
        CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_topk_sort_end, stream));
#endif

        // Top-k doc IDs are in d_sorted_doc_ids[0..actual_k_out-1]
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_d2d_start, stream));
#endif
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_topk_doc_ids, ws_.d_sorted_doc_ids,
                                   actual_k_out * sizeof(int), cudaMemcpyDeviceToDevice, stream));
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.s1_d2d_end, stream));
#endif

        // Top-k doc IDs now in ws_.d_topk_doc_ids[0..actual_k_out-1]
        // These remain on GPU for Stage 2 consumption

    }

    static constexpr int N_OVERLAP_CHUNKS = 4;   // Number of GPU scoring chunks
    int PRELIM_PER_CHUNK = k_rank_all_tokens / N_OVERLAP_CHUNKS;  // Max docs to CPU-refine per chunk

// ============================================================
// rank_stage23_persistent: fused Stage 2 + Stage 3 with streaming overlap
// ============================================================
    inline void rank_stage23_persistent(
        int num_candidates,
        size_t k,
        size_t k_stage2,        // k_rank_all_tokens (e.g., 300)
        query_object* queries,
        std::vector<size_t>& result
    ) {
        if (num_candidates == 0) return;

        auto cpu_ms = [](auto a, auto b) { return std::chrono::duration<float, std::milli>(b - a).count(); };

        cudaStream_t stream = ws_.stream_compute;
        cudaStream_t stream_d2h = ws_.stream_d2h;
        int actual_k = (int)std::min(k_stage2, (size_t)num_candidates);

#ifdef GPU_MVR_TIMELINE
        // --- Timeline instrumentation setup ---
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t tl_base;
        CUDA_CHECK(cudaEventCreate(&tl_base));
        CUDA_CHECK(cudaEventRecord(tl_base, stream));
        CUDA_CHECK(cudaEventSynchronize(tl_base));
        auto tl_base_cpu = std::chrono::high_resolution_clock::now();

        struct TLSpan { int start_us, end_us; std::string name, type; };
        std::vector<TLSpan> tl_gpu_compute, tl_gpu_d2h, tl_cpu;
        int tl_actual_chunks = 0; // set later when actual_chunks is known

        auto tl_cpu_us = [&](std::chrono::high_resolution_clock::time_point tp) -> int {
            return (int)std::chrono::duration_cast<std::chrono::microseconds>(tp - tl_base_cpu).count();
        };
        auto tl_gpu_us = [&](cudaEvent_t e) -> int {
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, tl_base, e)); return (int)(ms * 1000.0f);
        };

        // Phase C stream_d2h events (per-chunk)
        cudaEvent_t tl_c_wait_ev[N_OVERLAP_CHUNKS], tl_c_scores_s[N_OVERLAP_CHUNKS];
        cudaEvent_t tl_c_scores_e[N_OVERLAP_CHUNKS];
        cudaEvent_t tl_c_extract_s[N_OVERLAP_CHUNKS], tl_c_extract_m[N_OVERLAP_CHUNKS], tl_c_extract_e[N_OVERLAP_CHUNKS];
        bool tl_c_has_extract[N_OVERLAP_CHUNKS] = {};
        bool tl_c_has_scores[N_OVERLAP_CHUNKS] = {};
        for (int c = 0; c < N_OVERLAP_CHUNKS; c++) {
            CUDA_CHECK(cudaEventCreate(&tl_c_wait_ev[c]));
            CUDA_CHECK(cudaEventCreate(&tl_c_scores_s[c]));
            CUDA_CHECK(cudaEventCreate(&tl_c_scores_e[c]));
            CUDA_CHECK(cudaEventCreate(&tl_c_extract_s[c]));
            CUDA_CHECK(cudaEventCreate(&tl_c_extract_m[c]));
            CUDA_CHECK(cudaEventCreate(&tl_c_extract_e[c]));
        }
#endif

        // ========================================
        // Phase A: Data Preparation (GPU)
        // ========================================
        auto t_start_phase_a = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventRecord(ws_.phase_a_start_event, stream));
        float time_phase_a_gather = 0;
        float time_phase_a_prefix = 0;
        float time_phase_a_token_ids = 0;
        float time_phase_a_d2h = 0;

        auto t_cpu0 = std::chrono::high_resolution_clock::now();

        // 1. Gather document lengths
        int threads = 256;
        int blocks = (num_candidates + threads - 1) / threads;

        gather_doc_lengths_kernel<<<blocks, threads, 0, stream>>>(
            ws_.d_topk_doc_ids, d_doc_ptrs_,
            ws_.d_pair_doc_ids,   // reuse as doc_lengths buffer
            num_candidates
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ws_.phase_a_gather_done_event, stream));

        auto t_cpu1 = std::chrono::high_resolution_clock::now();

        // 2. Prefix sum -> candidate offsets
        CUDA_CHECK(cudaMemsetAsync(ws_.d_pst_candidate_offsets, 0, sizeof(size_t), stream));
        {
            thrust::device_ptr<int> len_ptr(ws_.d_pair_doc_ids);
            auto xform_iter = thrust::make_transform_iterator(len_ptr, cast_int_size_t());
            size_t scan_temp = ws_.cub_temp_storage_bytes;
            cub::DeviceScan::InclusiveScan(
                ws_.d_cub_temp_storage, scan_temp,
                xform_iter, ws_.d_pst_candidate_offsets + 1,
                thrust::plus<size_t>(), num_candidates, stream);
        }
        CUDA_CHECK(cudaEventRecord(ws_.phase_a_prefix_done_event, stream));

        auto t_cpu2 = std::chrono::high_resolution_clock::now();

        // 3. Gather token IDs
        gather_token_ids_kernel<<<num_candidates, 256, 0, stream>>>(
            ws_.d_topk_doc_ids,
            d_doc_ptrs_,
            ws_.d_pst_candidate_offsets,
            ws_.d_pst_token_ids,
            num_candidates
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ws_.phase_a_token_ids_done_event, stream));

        auto t_cpu3 = std::chrono::high_resolution_clock::now();

        // 4. Copy total_tokens, candidate offsets, and doc IDs to CPU
        size_t total_tokens;
        XFER_RECORD_BEGIN(stream);
        CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_pst_total_tokens, ws_.d_pst_candidate_offsets + num_candidates,
                                    sizeof(size_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_pst_candidate_offsets, ws_.d_pst_candidate_offsets,
                                    (num_candidates + 1) * sizeof(size_t),
                                    cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_pst_candidate_doc_ids, ws_.d_topk_doc_ids,
                                    num_candidates * sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
        XFER_RECORD_END(stream, sizeof(size_t) + (num_candidates + 1) * sizeof(size_t) + num_candidates * sizeof(int), false);
        CUDA_CHECK(cudaEventRecord(ws_.phase_a_d2h_done_event, stream));

        auto t_cpu4 = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaStreamSynchronize(stream));
        total_tokens = *ws_.h_pinned_pst_total_tokens;
        auto* h_candidate_doc_ids = ws_.h_pinned_pst_candidate_doc_ids;

        auto t_cpu5 = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaEventElapsedTime(&time_phase_a_gather, ws_.phase_a_start_event, ws_.phase_a_gather_done_event));
        CUDA_CHECK(cudaEventElapsedTime(&time_phase_a_prefix, ws_.phase_a_gather_done_event, ws_.phase_a_prefix_done_event));
        CUDA_CHECK(cudaEventElapsedTime(&time_phase_a_token_ids, ws_.phase_a_prefix_done_event, ws_.phase_a_token_ids_done_event));
        CUDA_CHECK(cudaEventElapsedTime(&time_phase_a_d2h, ws_.phase_a_token_ids_done_event, ws_.phase_a_d2h_done_event));
        float time_phase_a_gpu = 0;
        CUDA_CHECK(cudaEventElapsedTime(&time_phase_a_gpu, ws_.phase_a_start_event, ws_.phase_a_d2h_done_event));

        auto t_cpu6 = std::chrono::high_resolution_clock::now();
        float time_phase_a = std::chrono::duration<float, std::milli>(t_cpu6 - t_start_phase_a).count();

        // ========================================
        // Phase B: Launch ALL GPU chunk kernels on stream_compute
        //          D2H managed per-chunk in Phase C on stream_d2h
        // ========================================

        auto t_start_phase_b = std::chrono::high_resolution_clock::now();
        int score_threads = 128;
        while (score_threads < Q_DOCLEN && score_threads < 256) score_threads *= 2;
        score_threads = std::min(score_threads, 256);

        // Chunk boundaries (candidates are sorted by Stage 1 score descending)
        int actual_chunks = std::min((int)N_OVERLAP_CHUNKS, num_candidates);
        int cand_chunk_size = (num_candidates + actual_chunks - 1) / actual_chunks;
#ifdef GPU_MVR_TIMELINE
        tl_actual_chunks = actual_chunks;
#endif

        // Create per-chunk compute events
        cudaEvent_t chunk_compute_done[N_OVERLAP_CHUNKS];
        for (int c = 0; c < actual_chunks; c++)
            CUDA_CHECK(cudaEventCreateWithFlags(&chunk_compute_done[c], cudaEventDisableTiming));

#ifdef GPU_MVR_PROFILE
                cudaEvent_t chunk_bip_start[N_OVERLAP_CHUNKS];
                cudaEvent_t chunk_bip_end[N_OVERLAP_CHUNKS];
                cudaEvent_t chunk_docscore_start[N_OVERLAP_CHUNKS];
                cudaEvent_t chunk_docscore_end[N_OVERLAP_CHUNKS];
                bool chunk_has_bip[N_OVERLAP_CHUNKS] = {};
                ws_.s23_pst_kernel_ms = 0;
        ws_.s23_pst_bip_ms = 0;
                ws_.s23_pst_docscore_ms = 0;
                for (int c = 0; c < actual_chunks; c++) {
                        CUDA_CHECK(cudaEventCreate(&chunk_bip_start[c]));
                        CUDA_CHECK(cudaEventCreate(&chunk_bip_end[c]));
                        CUDA_CHECK(cudaEventCreate(&chunk_docscore_start[c]));
                        CUDA_CHECK(cudaEventCreate(&chunk_docscore_end[c]));
                }
                CUDA_CHECK(cudaEventRecord(ws_.s23_pst_kernel_start, stream));
#endif
        // Launch all chunks: binary_ip + doc_score → record compute event
        for (int c = 0; c < actual_chunks; c++) {
            int c_start = c * cand_chunk_size;
            int c_end = std::min(c_start + cand_chunk_size, num_candidates);
            int c_count = c_end - c_start;
            size_t tok_start = ws_.h_pinned_pst_candidate_offsets[c_start];
            size_t tok_end = ws_.h_pinned_pst_candidate_offsets[c_end];
            size_t tok_count = tok_end - tok_start;

            // binary_ip for this chunk's tokens
            if (tok_count > 0) {
                int bip_blocks = (tok_count + 255) / 256;
#ifdef GPU_MVR_PROFILE
                                chunk_has_bip[c] = true;
                                CUDA_CHECK(cudaEventRecord(chunk_bip_start[c], stream));
#endif
#ifdef GPU_MVR_USE_LUT
                {
                    size_t stage2_lut_smem = STAGE2_LUT_SMEM_FLOATS * sizeof(float) + STAGE2_LUT_TILE_Q * sizeof(float);
                    stage2_binary_ip_lut_kernel<<<bip_blocks, 256, stage2_lut_smem, stream>>>(
                        ws_.d_lut, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
                        ws_.d_pst_token_ids + tok_start,
                        ws_.d_token_dists + tok_start,
                        total_tokens, tok_count
                    );
                }
#else
                stage2_binary_ip_kernel_v2<<<bip_blocks, 256, 0, stream>>>(
                    ws_.d_queries, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
                    ws_.d_pst_token_ids + tok_start,
                    ws_.d_token_dists + tok_start,
                    total_tokens, tok_count
                );
#endif
#ifdef GPU_MVR_PROFILE
                                CUDA_CHECK(cudaEventRecord(chunk_bip_end[c], stream));
#endif
                CUDA_CHECK(cudaGetLastError());
            }

            // doc_score for this chunk's candidates
#ifdef GPU_MVR_PROFILE
                        CUDA_CHECK(cudaEventRecord(chunk_docscore_start[c], stream));
#endif
            doc_score_kernel<<<c_count, score_threads, 0, stream>>>(
                ws_.d_token_dists, ws_.d_pst_candidate_offsets + c_start,
                ws_.d_doc_scores + c_start, total_tokens, c_count
            );
#ifdef GPU_MVR_PROFILE
                        CUDA_CHECK(cudaEventRecord(chunk_docscore_end[c], stream));
#endif
            CUDA_CHECK(cudaGetLastError());

            // Record compute completion (gates D2H in Phase C)
            CUDA_CHECK(cudaEventRecord(chunk_compute_done[c], stream));
        }

#ifdef GPU_MVR_PROFILE
                CUDA_CHECK(cudaEventRecord(ws_.s23_pst_kernel_end, stream));
#endif

                // Record event after all GPU scoring
        CUDA_CHECK(cudaEventRecord(ws_.pst_extract_done, stream));
        auto t_end_phase_b = std::chrono::high_resolution_clock::now();
        float time_phase_b = std::chrono::duration<float, std::milli>(t_end_phase_b - t_start_phase_b).count();

        // ========================================
        // Phase C: Streaming CPU processing with GPU-extracted 1-bit dists
        //
        // Per chunk: D2H doc_scores → CPU top-k → GPU extract 1-bit dists
        // → CPU ip_ex_bits (overlapped with GPU extract) → combine → heap
        // ========================================

        auto t_start_phase_c = std::chrono::high_resolution_clock::now();
        double time_wait_d2h = 0;
        double time_running_topk = 0;
        double time_identify_new = 0;
        double time_gpu_extract = 0;
        double time_cpu_ip_ex = 0;
        double time_wait_extract = 0;
        double time_combine = 0;

        std::priority_queue<std::pair<float, size_t>> cpu_heap;
        std::unordered_set<int> seen_doc_ids;
        seen_doc_ids.reserve(actual_k * 2);
        int total_refined = 0;

        // Reusable buffers (storage pre-allocated in workspace)
        auto& running_indices = ws_.running_indices;
        auto& h_sel_indices   = ws_.h_sel_indices;
        auto& h_out_offsets   = ws_.h_out_offsets;
        // ip_ex_buf: intermediate storage for CPU-computed ex-bits inner products
        // Indexed as [flat_token_offset * Q_DOCLEN + query_idx]
        auto& ip_ex_buf       = ws_.ip_ex_buf;
        auto& refined_scores  = ws_.refined_scores;

        // Wake OMP worker threads now so they are hot when chunk 0's cpu_ip_ex fires.
        // Without this, threads sleep through the long Phase A+B gap and chunk 0
        // pays the full OS thread-wakeup latency (~100-200 ms) that chunks 1-3 avoid.
        #pragma omp parallel for schedule(static)
        for (int _i = 0; _i < 1; _i++) { (void)_i; }

        for (int c = 0; c < actual_chunks; c++) {
            // --- D2H doc_scores for this chunk ---
            auto t0 = std::chrono::high_resolution_clock::now();
            int c_start = c * cand_chunk_size;
            int c_end = std::min(c_start + cand_chunk_size, num_candidates);
            int c_count = c_end - c_start;
#ifdef GPU_MVR_TIMELINE
            CUDA_CHECK(cudaEventRecord(tl_c_wait_ev[c], stream_d2h));
#endif
            CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, chunk_compute_done[c], 0));
#ifdef GPU_MVR_TIMELINE
            CUDA_CHECK(cudaEventRecord(tl_c_scores_s[c], stream_d2h));
            tl_c_has_scores[c] = true;
#endif
            XFER_RECORD_BEGIN(stream_d2h);
            CUDA_CHECK(cudaMemcpyAsync(
                ws_.h_mapped_doc_scores + c_start,
                ws_.d_doc_scores + c_start,
                c_count * sizeof(float),
                cudaMemcpyDeviceToHost, stream_d2h));
            XFER_RECORD_END(stream_d2h, c_count * sizeof(float), false);
#ifdef GPU_MVR_TIMELINE
            CUDA_CHECK(cudaEventRecord(tl_c_scores_e[c], stream_d2h));
#endif
            CUDA_CHECK(cudaStreamSynchronize(stream_d2h));
            auto t1 = std::chrono::high_resolution_clock::now();
            time_wait_d2h += std::chrono::duration<double, std::milli>(t1 - t0).count();
#ifdef GPU_MVR_TIMELINE
            tl_cpu.push_back({tl_cpu_us(t0), tl_cpu_us(t1), "wait_d2h_c" + std::to_string(c), "async"});
#endif

            // --- CPU: running top-k from all scores seen so far [0, c_end) ---
            std::iota(running_indices.begin(), running_indices.begin() + c_end, 0);
            int run_k = std::min(actual_k * (c + 1) / actual_chunks, c_end);
            std::nth_element(running_indices.begin(), running_indices.begin() + run_k,
                            running_indices.begin() + c_end,
                            [this](int a, int b) {
                                return ws_.h_mapped_doc_scores[a] > ws_.h_mapped_doc_scores[b];
                            });
            auto t2 = std::chrono::high_resolution_clock::now();
            time_running_topk += std::chrono::duration<double, std::milli>(t2 - t1).count();
#ifdef GPU_MVR_TIMELINE
            tl_cpu.push_back({tl_cpu_us(t1), tl_cpu_us(t2), "topk_c" + std::to_string(c), "async"});
#endif

            // --- Identify NEW docs in running top-k (not yet refined) ---
            std::vector<std::pair<float, int>> new_docs;
            new_docs.reserve(run_k);
            for (int i = 0; i < run_k; i++) {
                int cand_idx = running_indices[i];
                int doc_id = h_candidate_doc_ids[cand_idx];
                if (seen_doc_ids.find(doc_id) == seen_doc_ids.end()) {
                    new_docs.emplace_back(ws_.h_mapped_doc_scores[cand_idx], cand_idx);
                }
            }

            if (new_docs.empty()) {
#ifdef GPU_MVR_TIMELINE
                auto t_skip = std::chrono::high_resolution_clock::now();
                tl_cpu.push_back({tl_cpu_us(t2), tl_cpu_us(t_skip), "identify_c" + std::to_string(c), "async"});
#endif
                continue;
            }
            std::sort(new_docs.begin(), new_docs.end(), std::greater<>());
            int to_refine = new_docs.size();
            new_docs.resize(to_refine);
            for (const auto& p : new_docs) {
                seen_doc_ids.insert(h_candidate_doc_ids[p.second]);
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            time_identify_new += std::chrono::duration<double, std::milli>(t3 - t2).count();
#ifdef GPU_MVR_TIMELINE
            tl_cpu.push_back({tl_cpu_us(t2), tl_cpu_us(t3), "identify_c" + std::to_string(c), "async"});
#endif

            // --- Prepare selected indices + output offsets for GPU extract ---
            h_out_offsets[0] = 0;
            for (int i = 0; i < to_refine; i++) {
                h_sel_indices[i] = new_docs[i].second;  // candidate index
                int doc_id = h_candidate_doc_ids[h_sel_indices[i]];
                h_out_offsets[i + 1] = h_out_offsets[i] +
                    (size_t)(doc_ptrs_[doc_id + 1] - doc_ptrs_[doc_id]);
            }
            size_t total_sel_tokens = h_out_offsets[to_refine];

            // --- Launch GPU extract + D2H (async on stream_d2h) ---
            // H2D of metadata (tiny, synchronous due to pageable memory — negligible)
#ifdef GPU_MVR_TIMELINE
            CUDA_CHECK(cudaEventRecord(tl_c_extract_s[c], stream_d2h));
            tl_c_has_extract[c] = true;
#endif
            XFER_RECORD_BEGIN(stream_d2h);
            CUDA_CHECK(cudaMemcpyAsync(ws_.d_selected_indices, h_sel_indices.data(),
                                        to_refine * sizeof(int),
                                        cudaMemcpyHostToDevice, stream_d2h));
            CUDA_CHECK(cudaMemcpyAsync(ws_.d_out_offsets, h_out_offsets.data(),
                                        (to_refine + 1) * sizeof(size_t),
                                        cudaMemcpyHostToDevice, stream_d2h));
            XFER_RECORD_END(stream_d2h, to_refine * sizeof(int) + (to_refine + 1) * sizeof(size_t), true);
            // Extract kernel: compacts scattered 1-bit dists into contiguous output
            extract_one_bit_dists_kernel<<<to_refine, 256, 0, stream_d2h>>>(
                ws_.d_token_dists, ws_.d_pst_candidate_offsets,
                ws_.d_selected_indices, ws_.d_out_one_bit_dists,
                ws_.d_out_offsets, total_tokens, (size_t)to_refine
            );
            CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_TIMELINE
            CUDA_CHECK(cudaEventRecord(tl_c_extract_m[c], stream_d2h));
#endif
            // D2H extracted dists (contiguous, async — pinned target)
            XFER_RECORD_BEGIN(stream_d2h);
            CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_dists, ws_.d_out_one_bit_dists,
                                        total_sel_tokens * Q_DOCLEN * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream_d2h));
            XFER_RECORD_END(stream_d2h, total_sel_tokens * Q_DOCLEN * sizeof(float), false);
            // extract_one_bit_dists_kernel_v2<<<1, 384, 0, stream_d2h>>>(
            //     ws_.d_token_dists, ws_.d_pst_candidate_offsets,
            //     ws_.d_selected_indices, ws_.d_pinned_dists,
            //     ws_.d_out_offsets, total_tokens, (size_t)to_refine
            // );
            // CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_TIMELINE
            CUDA_CHECK(cudaEventRecord(tl_c_extract_e[c], stream_d2h));
#endif
            auto t4 = std::chrono::high_resolution_clock::now();
            time_gpu_extract += std::chrono::duration<double, std::milli>(t4 - t3).count();
#ifdef GPU_MVR_TIMELINE
            tl_cpu.push_back({tl_cpu_us(t3), tl_cpu_us(t4), "launch_extract_c" + std::to_string(c), "async"});
#endif

            // --- CPU: compute ip_ex_bits (OVERLAPPED with GPU extract + D2H) ---
            cpu_compute_ip_ex(
                h_candidate_doc_ids, h_sel_indices.data(), h_out_offsets.data(),
                to_refine, ws_.h_pinned_queries, ip_ex_buf.data());
            auto t5 = std::chrono::high_resolution_clock::now();
            time_cpu_ip_ex += std::chrono::duration<double, std::milli>(t5 - t4).count();
#ifdef GPU_MVR_TIMELINE
            tl_cpu.push_back({tl_cpu_us(t4), tl_cpu_us(t5), "cpu_ip_ex_c" + std::to_string(c), "static"});
#endif

            // --- Wait for GPU extract D2H to complete ---
            CUDA_CHECK(cudaStreamSynchronize(stream_d2h));
            auto t6 = std::chrono::high_resolution_clock::now();
            time_wait_extract += std::chrono::duration<double, std::milli>(t6 - t5).count();
#ifdef GPU_MVR_TIMELINE
            tl_cpu.push_back({tl_cpu_us(t5), tl_cpu_us(t6), "wait_extract_c" + std::to_string(c), "async"});
#endif

            // --- Combine: GPU 1-bit dists + CPU ip_ex_bits → final scores ---
            cpu_combine_scores(
                h_candidate_doc_ids, h_sel_indices.data(), h_out_offsets.data(),
                to_refine, queries, ws_.h_pinned_dists, ip_ex_buf.data(),
                refined_scores.data());

            // Add refined results to heap
            for (int i = 0; i < to_refine; i++) {
                cpu_heap.emplace(refined_scores[i].first, (size_t)refined_scores[i].second);
            }
            auto t7 = std::chrono::high_resolution_clock::now();
            time_combine += std::chrono::duration<double, std::milli>(t7 - t6).count();
#ifdef GPU_MVR_TIMELINE
            tl_cpu.push_back({tl_cpu_us(t6), tl_cpu_us(t7), "combine_c" + std::to_string(c), "subflow"});
#endif

            total_refined += to_refine;
        }

        auto t_end_phase_c = std::chrono::high_resolution_clock::now();
        double time_phase_c = std::chrono::duration<double, std::milli>(t_end_phase_c - t_start_phase_c).count();
        double total_wall_time = std::chrono::duration<double, std::milli>(t_end_phase_c - t_start_phase_a).count();

#ifdef GPU_MVR_PROFILE
                CUDA_CHECK(cudaEventSynchronize(ws_.s23_pst_kernel_end));
                CUDA_CHECK(cudaEventElapsedTime(&ws_.s23_pst_kernel_ms, ws_.s23_pst_kernel_start, ws_.s23_pst_kernel_end));
                for (int c = 0; c < actual_chunks; c++) {
                        if (chunk_has_bip[c]) {
                                float chunk_bip_ms = 0;
                                CUDA_CHECK(cudaEventElapsedTime(&chunk_bip_ms, chunk_bip_start[c], chunk_bip_end[c]));
                                ws_.s23_pst_bip_ms += chunk_bip_ms;
                        }

                        float chunk_docscore_ms = 0;
                        CUDA_CHECK(cudaEventElapsedTime(&chunk_docscore_ms, chunk_docscore_start[c], chunk_docscore_end[c]));
                        ws_.s23_pst_docscore_ms += chunk_docscore_ms;
                }
#endif

        // ========================================
        // Phase D: Extract final top-k results (trivial)
        // ========================================
        result.clear();
        for (size_t i = 0; i < k && !cpu_heap.empty(); ++i) {
            result.push_back(cpu_heap.top().second);
            cpu_heap.pop();
        }

        // Cleanup compute events
        for (int c = 0; c < actual_chunks; c++)
            CUDA_CHECK(cudaEventDestroy(chunk_compute_done[c]));

#ifdef GPU_MVR_TIMELINE
        // ========================================
        // Timeline JSON output (must run before GPU_MVR_PROFILE event cleanup)
        // ========================================
        {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaStreamSynchronize(stream_d2h));

            // --- Build GPU compute spans (Phase A on stream_compute) ---
            tl_gpu_compute.push_back({tl_gpu_us(ws_.phase_a_start_event), tl_gpu_us(ws_.phase_a_gather_done_event), "gather_doc_lengths", "async"});
            tl_gpu_compute.push_back({tl_gpu_us(ws_.phase_a_gather_done_event), tl_gpu_us(ws_.phase_a_prefix_done_event), "prefix_sum", "async"});
            tl_gpu_compute.push_back({tl_gpu_us(ws_.phase_a_prefix_done_event), tl_gpu_us(ws_.phase_a_token_ids_done_event), "gather_token_ids", "async"});
            tl_gpu_compute.push_back({tl_gpu_us(ws_.phase_a_token_ids_done_event), tl_gpu_us(ws_.phase_a_d2h_done_event), "d2h_metadata", "async"});

            // --- Build GPU compute spans (Phase B on stream_compute) ---
#ifdef GPU_MVR_PROFILE
            for (int c = 0; c < tl_actual_chunks; c++) {
                if (chunk_has_bip[c]) {
                    tl_gpu_compute.push_back({tl_gpu_us(chunk_bip_start[c]), tl_gpu_us(chunk_bip_end[c]),
                        "binary_ip_c" + std::to_string(c), "static"});
                }
                tl_gpu_compute.push_back({tl_gpu_us(chunk_docscore_start[c]), tl_gpu_us(chunk_docscore_end[c]),
                    "docscore_c" + std::to_string(c), "subflow"});
            }
#endif

            // --- Build GPU d2h spans (Phase C on stream_d2h) ---
            for (int c = 0; c < tl_actual_chunks; c++) {
                if (tl_c_has_scores[c]) {
                    tl_gpu_d2h.push_back({tl_gpu_us(tl_c_wait_ev[c]), tl_gpu_us(tl_c_scores_s[c]),
                        "wait_compute_c" + std::to_string(c), "async"});
                    tl_gpu_d2h.push_back({tl_gpu_us(tl_c_scores_s[c]), tl_gpu_us(tl_c_scores_e[c]),
                        "d2h_scores_c" + std::to_string(c), "condition"});
                }
                if (tl_c_has_extract[c]) {
                    tl_gpu_d2h.push_back({tl_gpu_us(tl_c_extract_s[c]), tl_gpu_us(tl_c_extract_m[c]),
                        "extract_kernel_c" + std::to_string(c), "module"});
                    tl_gpu_d2h.push_back({tl_gpu_us(tl_c_extract_m[c]), tl_gpu_us(tl_c_extract_e[c]),
                        "extract_d2h_c" + std::to_string(c), "condition"});
                }
            }

            // --- Build CPU spans (Phase A + B) ---
            tl_cpu.push_back({tl_cpu_us(t_cpu0), tl_cpu_us(t_cpu1), "launch_gather", "async"});
            tl_cpu.push_back({tl_cpu_us(t_cpu1), tl_cpu_us(t_cpu2), "launch_prefix", "async"});
            tl_cpu.push_back({tl_cpu_us(t_cpu2), tl_cpu_us(t_cpu3), "launch_token_ids", "async"});
            tl_cpu.push_back({tl_cpu_us(t_cpu3), tl_cpu_us(t_cpu4), "launch_d2h", "async"});
            tl_cpu.push_back({tl_cpu_us(t_cpu4), tl_cpu_us(t_cpu5), "sync_stream", "async"});
            tl_cpu.push_back({tl_cpu_us(t_cpu5), tl_cpu_us(t_cpu6), "event_elapsed", "async"});
            tl_cpu.push_back({tl_cpu_us(t_start_phase_b), tl_cpu_us(t_end_phase_b), "launch_phase_b", "async"});
            // Phase C CPU spans were already pushed inline in the loop

            // Sort all span lists by start time
            auto span_cmp = [](const TLSpan& a, const TLSpan& b) { return a.start_us < b.start_us; };
            std::sort(tl_gpu_compute.begin(), tl_gpu_compute.end(), span_cmp);
            std::sort(tl_gpu_d2h.begin(), tl_gpu_d2h.end(), span_cmp);
            std::sort(tl_cpu.begin(), tl_cpu.end(), span_cmp);

            // --- Write JSON ---
            auto write_spans = [](std::ostringstream& os, const std::vector<TLSpan>& spans) {
                for (size_t i = 0; i < spans.size(); i++) {
                    if (i > 0) os << ",";
                    os << "{\"span\":[" << spans[i].start_us << "," << spans[i].end_us
                       << "],\"name\":\"" << spans[i].name
                       << "\",\"type\":\"" << spans[i].type << "\"}";
                }
            };

            std::ostringstream oss;
            oss << "[{\"executor\":0,\"data\":[";
            // Worker 1: stream_compute
            oss << "{\"worker\":1,\"level\":0,\"data\":[";
            write_spans(oss, tl_gpu_compute);
            oss << "]},";
            // Worker 2: stream_d2h
            oss << "{\"worker\":2,\"level\":0,\"data\":[";
            write_spans(oss, tl_gpu_d2h);
            oss << "]}]},";
            // Executor 1: CPU
            oss << "{\"executor\":1,\"data\":[";
            oss << "{\"worker\":1,\"level\":0,\"data\":[";
            write_spans(oss, tl_cpu);
            oss << "]}]}]";

            std::ofstream tlf("timeline.json");
            tlf << oss.str();
            tlf.close();
            std::cout << "[TIMELINE] Wrote timeline.json\n";

            // Cleanup timeline events
            CUDA_CHECK(cudaEventDestroy(tl_base));
            for (int c = 0; c < N_OVERLAP_CHUNKS; c++) {
                CUDA_CHECK(cudaEventDestroy(tl_c_wait_ev[c]));
                CUDA_CHECK(cudaEventDestroy(tl_c_scores_s[c]));
                CUDA_CHECK(cudaEventDestroy(tl_c_scores_e[c]));
                CUDA_CHECK(cudaEventDestroy(tl_c_extract_s[c]));
                CUDA_CHECK(cudaEventDestroy(tl_c_extract_m[c]));
                CUDA_CHECK(cudaEventDestroy(tl_c_extract_e[c]));
            }
        }
#endif

#ifdef GPU_MVR_PROFILE
        std::cout << "[PROFILE] Phase A (Data Preparation) wall: " << time_phase_a << " ms, GPU total: " << time_phase_a_gpu << " ms\n";
        std::cout << "[PROFILE]   1. Gather doc lengths      : " << time_phase_a_gather << " ms (CPU launch: " << cpu_ms(t_cpu0, t_cpu1) << " ms)\n";
        std::cout << "[PROFILE]   2. Prefix sum offsets      : " << time_phase_a_prefix << " ms (CPU launch: " << cpu_ms(t_cpu1, t_cpu2) << " ms)\n";
        std::cout << "[PROFILE]   3. Gather token IDs        : " << time_phase_a_token_ids << " ms (CPU launch: " << cpu_ms(t_cpu2, t_cpu3) << " ms)\n";
        std::cout << "[PROFILE]   4. D2H metadata + sync     : " << time_phase_a_d2h << " ms (CPU launch: " << cpu_ms(t_cpu3, t_cpu4) << " ms)\n";
        std::cout << "[PROFILE]   5. cudaStreamSynchronize   : " << cpu_ms(t_cpu4, t_cpu5) << " ms\n";
        std::cout << "[PROFILE]   6. cudaEventElapsedTime x5 : " << cpu_ms(t_cpu5, t_cpu6) << " ms\n";
        std::cout << "[PROFILE]   GPU sum accounted           : "
                  << (time_phase_a_gather + time_phase_a_prefix + time_phase_a_token_ids + time_phase_a_d2h)
                  << " ms\n";
        std::cout << "[PROFILE] Phase B wall time: " << time_phase_b << " ms\n";
        printf("[PROFILE] Phase C: wait_d2h=%.3f ms, topk=%.3f ms, identify=%.3f ms, "
            "gpu_extract=%.3f ms, cpu_ip_ex=%.3f ms, wait_extract=%.3f ms, "
            "combine=%.3f ms (total=%.3f ms, %d docs)\n",
            time_wait_d2h, time_running_topk, time_identify_new,
            time_gpu_extract, time_cpu_ip_ex, time_wait_extract,
            time_combine, time_phase_c, total_refined);
        std::cout << "[PROFILE] Total wall time for Phase A + B + C: " << total_wall_time << " ms\n";
        for (int c = 0; c < actual_chunks; c++) {
            CUDA_CHECK(cudaEventDestroy(chunk_bip_start[c]));
            CUDA_CHECK(cudaEventDestroy(chunk_bip_end[c]));
            CUDA_CHECK(cudaEventDestroy(chunk_docscore_start[c]));
            CUDA_CHECK(cudaEventDestroy(chunk_docscore_end[c]));
        }
#endif
    }

    // ======================== STAGE 3: CPU ========================

    // Compute ex-bits inner products for a batch of documents.
    // For each doc's tokens, unpacks the compact ex-code and computes dot products
    // against all queries via batched GEMV (AVX-512).
    //
    // h_sel_indices[0..to_refine): candidate indices into h_candidate_doc_ids
    // h_out_offsets[0..to_refine]: prefix-sum of token counts per doc
    // ip_ex_buf: output buffer indexed as [(h_out_offsets[i] + t) * Q_DOCLEN + q]
    inline void cpu_compute_ip_ex(
        const int* h_candidate_doc_ids,
        const int* h_sel_indices,
        const size_t* h_out_offsets,
        int to_refine,
        const float* queries_flat,
        float* ip_ex_buf
    ) {
        const size_t ex_code_stride = PADDED_DIM * ex_bits / 8;
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < to_refine; i++) {
            int doc_id = h_candidate_doc_ids[h_sel_indices[i]];
            size_t doc_start = doc_ptrs_[doc_id];
            size_t n_tok = doc_ptrs_[doc_id + 1] - doc_start;
            alignas(64) float decoded[PADDED_DIM];
            for (size_t t = 0; t < n_tok; t++) {
                size_t tid = doc_start + t;
                unpack_func_(
                    reinterpret_cast<const uint8_t*>(&ex_code_[tid * ex_code_stride]),
                    decoded, PADDED_DIM);
                if (t + 1 < n_tok)
                    __builtin_prefetch(&ex_code_[(tid + 1) * ex_code_stride], 0, 3);
                rabitqlib::gemv_batch8_avx512(
                    queries_flat, decoded,
                    &ip_ex_buf[(h_out_offsets[i] + t) * Q_DOCLEN],
                    Q_DOCLEN, PADDED_DIM);
            }
        }
    }

    // Combine GPU 1-bit dists + CPU ex-bits IPs → final doc scores.
    // For each doc, computes per-query max-over-tokens of the combined distance,
    // then sums across queries to get the final doc score.
    //
    // h_pinned_dists: GPU-extracted 1-bit dists [(h_out_offsets[i]+t)*Q_DOCLEN + q]
    // ip_ex_buf:      CPU-computed ex-bits IPs, same layout
    // refined_scores: output pairs (score, doc_id) for each doc
    inline void cpu_combine_scores(
        const int* h_candidate_doc_ids,
        const int* h_sel_indices,
        const size_t* h_out_offsets,
        int to_refine,
        const query_object* queries,
        const float* h_pinned_dists,
        const float* ip_ex_buf,
        std::pair<float, int>* refined_scores
    ) {
        const float scale = static_cast<float>(1 << ex_bits);
        // Precompute per-query bias: scale * cb1_sumq[j] - cbex_sumq[j]
        alignas(64) float query_bias[Q_DOCLEN];
        for (size_t j = 0; j < Q_DOCLEN; j++)
            query_bias[j] = scale * queries[j].cb1_sumq - queries[j].cbex_sumq;

#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < to_refine; i++) {
            int doc_id = h_candidate_doc_ids[h_sel_indices[i]];
            size_t doc_start = doc_ptrs_[doc_id];
            size_t n_tok = doc_ptrs_[doc_id + 1] - doc_start;

            // Per-query running max, initialized to -inf
            alignas(64) float max_ts[Q_DOCLEN];
            __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
            for (size_t j = 0; j < Q_DOCLEN; j += 16)
                _mm512_store_ps(&max_ts[j], neg_inf);

            for (size_t t = 0; t < n_tok; t++) {
                size_t tid = doc_start + t;
                float tok_scale = scale / one_bit_factor_[tid];
                float tok_exf   = ex_factor_[tid];
                __m512 v_tok_scale = _mm512_set1_ps(tok_scale);
                __m512 v_tok_exf   = _mm512_set1_ps(tok_exf);

                const float* ob_base = &h_pinned_dists[(h_out_offsets[i] + t) * Q_DOCLEN];
                const float* ex_base = &ip_ex_buf[(h_out_offsets[i] + t) * Q_DOCLEN];

                for (size_t j = 0; j < Q_DOCLEN; j += 16) {
                    __m512 ob  = _mm512_loadu_ps(&ob_base[j]);
                    __m512 exd = _mm512_loadu_ps(&ex_base[j]);
                    __m512 qb  = _mm512_load_ps(&query_bias[j]);
                    // combined = (tok_scale * ob + qb + exd) * tok_exf
                    __m512 combined = _mm512_mul_ps(
                        _mm512_add_ps(_mm512_fmadd_ps(v_tok_scale, ob, qb), exd),
                        v_tok_exf);
                    _mm512_store_ps(&max_ts[j],
                        _mm512_max_ps(_mm512_load_ps(&max_ts[j]), combined));
                }
            }

            // Sum per-query maxes → doc_score
            __m512 sum = _mm512_load_ps(&max_ts[0]);
            for (size_t j = 16; j < Q_DOCLEN; j += 16)
                sum = _mm512_add_ps(sum, _mm512_load_ps(&max_ts[j]));
            refined_scores[i] = {_mm512_reduce_add_ps(sum), doc_id};
        }
    }

    inline void cpu_compute_ip_ex_tbb(
        const int* h_candidate_doc_ids,
        const int* h_sel_indices,
        const size_t* h_out_offsets,
        int to_refine,
        const float* queries_flat,
        float* ip_ex_buf
    ) {
        const size_t ex_code_stride = PADDED_DIM * ex_bits / 8;

        // Grain sizes: tune these!
        const int GRAIN_I = 1;     // small because to_refine is small
        const int GRAIN_T = 32;    // key: creates enough parallelism

        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range2d<int, int>(
                0, to_refine, GRAIN_I,
                0, max_doc_len, GRAIN_T
            ),
            [&](const oneapi::tbb::blocked_range2d<int, int>& r) {

                for (int i = r.rows().begin(); i < r.rows().end(); ++i) {

                    int doc_id = h_candidate_doc_ids[h_sel_indices[i]];
                    size_t doc_start = doc_ptrs_[doc_id];
                    size_t n_tok = doc_ptrs_[doc_id + 1] - doc_start;

                    // clamp token range per doc
                    int t_begin = std::min<int>(r.cols().begin(), n_tok);
                    int t_end   = std::min<int>(r.cols().end(),   n_tok);

                    alignas(64) float decoded[PADDED_DIM];

                    for (int t = t_begin; t < t_end; ++t) {
                        size_t tid = doc_start + t;

                        unpack_func_(
                            reinterpret_cast<const uint8_t*>(&ex_code_[tid * ex_code_stride]),
                            decoded, PADDED_DIM);

                        if (t + 1 < n_tok)
                            __builtin_prefetch(&ex_code_[(tid + 1) * ex_code_stride], 0, 3);

                        rabitqlib::gemv_batch8_avx512(
                            queries_flat,
                            decoded,
                            &ip_ex_buf[(h_out_offsets[i] + t) * Q_DOCLEN],
                            Q_DOCLEN,
                            PADDED_DIM);
                    }
                }
            }
        );
    }

    inline void cpu_combine_scores_tbb(
        const int* h_candidate_doc_ids,
        const int* h_sel_indices,
        const size_t* h_out_offsets,
        int to_refine,
        const query_object* queries,
        const float* h_pinned_dists,
        const float* ip_ex_buf,
        std::pair<float, int>* refined_scores
    ) {
        const float scale = static_cast<float>(1 << ex_bits);

        alignas(64) float query_bias[Q_DOCLEN];
        for (size_t j = 0; j < Q_DOCLEN; j++)
            query_bias[j] = scale * queries[j].cb1_sumq - queries[j].cbex_sumq;

        const int GRAIN_I = 1;
        const int GRAIN_T = 32;

        oneapi::tbb::parallel_for(0, to_refine, GRAIN_I, [&](int i) {

            int doc_id = h_candidate_doc_ids[h_sel_indices[i]];
            size_t doc_start = doc_ptrs_[doc_id];
            size_t n_tok = doc_ptrs_[doc_id + 1] - doc_start;

            alignas(64) float max_ts[Q_DOCLEN];

            __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
            for (size_t j = 0; j < Q_DOCLEN; j += 16)
                _mm512_store_ps(&max_ts[j], neg_inf);

            for (size_t t = 0; t < n_tok; t++) {
                size_t tid = doc_start + t;

                float tok_scale = scale / one_bit_factor_[tid];
                float tok_exf   = ex_factor_[tid];

                __m512 v_tok_scale = _mm512_set1_ps(tok_scale);
                __m512 v_tok_exf   = _mm512_set1_ps(tok_exf);

                const float* ob_base = &h_pinned_dists[(h_out_offsets[i] + t) * Q_DOCLEN];
                const float* ex_base = &ip_ex_buf[(h_out_offsets[i] + t) * Q_DOCLEN];

                for (size_t j = 0; j < Q_DOCLEN; j += 16) {
                    __m512 ob  = _mm512_loadu_ps(&ob_base[j]);
                    __m512 exd = _mm512_loadu_ps(&ex_base[j]);
                    __m512 qb  = _mm512_load_ps(&query_bias[j]);

                    __m512 combined = _mm512_mul_ps(
                        _mm512_add_ps(_mm512_fmadd_ps(v_tok_scale, ob, qb), exd),
                        v_tok_exf);

                    _mm512_store_ps(&max_ts[j],
                        _mm512_max_ps(_mm512_load_ps(&max_ts[j]), combined));
                }
            }

            __m512 sum = _mm512_load_ps(&max_ts[0]);
            for (size_t j = 16; j < Q_DOCLEN; j += 16)
                sum = _mm512_add_ps(sum, _mm512_load_ps(&max_ts[j]));

            refined_scores[i] = {_mm512_reduce_add_ps(sum), doc_id};
        });
    }

    ~gpu_mvr_index() {
        // Free persistent GPU data
        CUDA_CHECK(cudaFree(d_one_bit_code_));
        CUDA_CHECK(cudaFree(d_one_bit_factor_));
        CUDA_CHECK(cudaFree(d_doc_ids_));
        CUDA_CHECK(cudaFree(d_doc_ptrs_));
        CUDA_CHECK(cudaFree(d_inv_list_));
        CUDA_CHECK(cudaFree(d_cluster_pos_));

        // Free workspace
        CUDA_CHECK(cudaFree(ws_.d_queries));
        CUDA_CHECK(cudaFree(ws_.d_cb1_sumq));
#ifdef GPU_MVR_USE_LUT
        CUDA_CHECK(cudaFree(ws_.d_lut));
#endif
        CUDA_CHECK(cudaFree(ws_.d_emb_ids));
        CUDA_CHECK(cudaFree(ws_.d_pair_offsets));
        CUDA_CHECK(cudaFree(ws_.d_emb_dists));
        CUDA_CHECK(cudaFree(ws_.d_pair_doc_ids));
        CUDA_CHECK(cudaFree(ws_.d_pair_query_indices));
#ifndef GPU_MVR_OVERLAP_STAGE23
        CUDA_CHECK(cudaFree(ws_.d_token_ids));
        CUDA_CHECK(cudaFree(ws_.d_candidate_offsets));
        CUDA_CHECK(cudaFree(ws_.d_token_dists));
        CUDA_CHECK(cudaFree(ws_.d_doc_scores));
#endif
        // Extract workspace (both paths)
        CUDA_CHECK(cudaFree(ws_.d_selected_indices));
        CUDA_CHECK(cudaFree(ws_.d_out_offsets));
        CUDA_CHECK(cudaFree(ws_.d_out_one_bit_dists));

        CUDA_CHECK(cudaFree(ws_.d_cagra_dists));
        CUDA_CHECK(cudaFree(ws_.d_cagra_labels));

        // === NEW: Free GPU-only aggregation workspace ===
        CUDA_CHECK(cudaFree(ws_.d_sorted_doc_ids));
        CUDA_CHECK(cudaFree(ws_.d_sorted_query_indices));
        CUDA_CHECK(cudaFree(ws_.d_sorted_dists));
        CUDA_CHECK(cudaFree(ws_.d_unique_doc_ids));
        CUDA_CHECK(cudaFree(ws_.d_doc_offsets));
        CUDA_CHECK(cudaFree(ws_.d_stage1_doc_scores));
        CUDA_CHECK(cudaFree(ws_.d_doc_query_max));
        CUDA_CHECK(cudaFree(ws_.d_num_unique_docs));
        CUDA_CHECK(cudaFree(ws_.d_cub_temp_storage));
        CUDA_CHECK(cudaFree(ws_.d_topk_scores));
        CUDA_CHECK(cudaFree(ws_.d_topk_doc_ids));
        CUDA_CHECK(cudaFree(ws_.d_topk_indices));

        // Free pinned memory
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_queries));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_cb1_sumq));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_dists));  // also frees the mapped device pointer
#ifndef GPU_MVR_OVERLAP_STAGE23
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_batch_scores));
#endif

#ifdef GPU_MVR_OVERLAP_STAGE23
        // === Persistent Stage 2+3 workspace ===
        CUDA_CHECK(cudaFree(ws_.d_token_dists));
        CUDA_CHECK(cudaFree(ws_.d_pst_candidate_offsets));
        CUDA_CHECK(cudaFree(ws_.d_pst_token_ids));
        CUDA_CHECK(cudaFree(ws_.d_doc_scores));
        CUDA_CHECK(cudaFreeHost(ws_.h_mapped_doc_scores));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_pst_candidate_offsets));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_pst_candidate_doc_ids));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_pst_total_tokens));
        CUDA_CHECK(cudaEventDestroy(ws_.pst_compute_done));
        CUDA_CHECK(cudaEventDestroy(ws_.pst_extract_done));
        for (int i = 0; i < Workspace::PST_NUM_D2H_CHUNKS; i++)
            CUDA_CHECK(cudaEventDestroy(ws_.pst_d2h_chunk_done[i]));
#endif

        // === NEW: Destroy streams and events ===
        CUDA_CHECK(cudaStreamDestroy(ws_.stream_compute));
        CUDA_CHECK(cudaStreamDestroy(ws_.stream_h2d));
        CUDA_CHECK(cudaStreamDestroy(ws_.stream_d2h));
        CUDA_CHECK(cudaEventDestroy(ws_.event_h2d_done));

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventDestroy(ws_.event_start));
        CUDA_CHECK(cudaEventDestroy(ws_.event_end));
        CUDA_CHECK(cudaEventDestroy(ws_.event_stage1_start));
        CUDA_CHECK(cudaEventDestroy(ws_.event_stage1_end));
        CUDA_CHECK(cudaEventDestroy(ws_.event_stage2_start));
        CUDA_CHECK(cudaEventDestroy(ws_.event_stage2_end));

        // Stage 1 fine-grained kernel events
        CUDA_CHECK(cudaEventDestroy(ws_.s1_cagra_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_cagra_end));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_expansion_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_expansion_end));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_binary_ip_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_binary_ip_end));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_memset_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_memset_end));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_atomic_agg_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_atomic_agg_end));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_sum_scores_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_sum_scores_end));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_topk_sort_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_topk_sort_end));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_d2d_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s1_d2d_end));

        CUDA_CHECK(cudaEventDestroy(ws_.s2_gather_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s2_gather_end));
        CUDA_CHECK(cudaEventDestroy(ws_.s2_prefix_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s2_prefix_end));
        CUDA_CHECK(cudaEventDestroy(ws_.s2_tokenids_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s2_tokenids_end));
        CUDA_CHECK(cudaEventDestroy(ws_.s2_binaryip_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s2_binaryip_end));
        CUDA_CHECK(cudaEventDestroy(ws_.s2_docscore_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s2_docscore_end));
        CUDA_CHECK(cudaEventDestroy(ws_.s2_d2h_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s2_d2h_end));
        CUDA_CHECK(cudaEventDestroy(ws_.s2_extract_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s2_extract_end));

        // Persistent Stage 2+3 profiling events
        CUDA_CHECK(cudaEventDestroy(ws_.s23_pst_kernel_start));
        CUDA_CHECK(cudaEventDestroy(ws_.s23_pst_kernel_end));

        // Transfer profiling events
        for (int i = 0; i < Workspace::MAX_XFER_RECORDS; i++) {
            CUDA_CHECK(cudaEventDestroy(ws_.xfer_records[i].start));
            CUDA_CHECK(cudaEventDestroy(ws_.xfer_records[i].end));
        }
#endif

        delete rotator_;
        delete ivf;
    }
};
