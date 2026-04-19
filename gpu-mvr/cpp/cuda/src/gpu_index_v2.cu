#include "gpu_index_v2.cuh"

#include <cub/cub.cuh>
#include <thrust/adjacent_difference.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

// Prevent Eigen from adding __host__ __device__ annotations to its functions.
// Eigen is only used on the CPU side; without this, nvcc flags warning #20014-D
// for Eigen internals that call host-only STL functions.
#define EIGEN_NO_CUDA

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <queue>
#include <stdexcept>
#include <unordered_set>

#include "estimator.hpp"
#include "gpu_kernels_v2.cuh"
#include "gpu_profiler.cuh"
#include "quantization.hpp"
#include "query.hpp"
#include "rabitqlib/quantization/pack_excode.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "gpu_index_layout.hpp"
#include "mvr_index_file_format.hpp"
#include "startup_profile.hpp"

using namespace rabitqlib;

namespace {

void validate_doc4ex_header_compatibility(
    const mvr_index_file_format::Header& base_header,
    const mvr_index_file_format::Header& ex_header,
    const std::string& base_filename,
    const std::string& ex_filename)
{
    if (base_header.n != ex_header.n ||
        base_header.d != ex_header.d ||
        base_header.n_clusters != ex_header.n_clusters ||
        base_header.padded_dim != ex_header.padded_dim ||
        base_header.rotator_type != ex_header.rotator_type) {
        throw std::runtime_error(
            "Quantized index metadata mismatch between " + base_filename +
            " and " + ex_filename);
    }
    if (ex_header.ex_bits == 0) {
        throw std::runtime_error(
            "doc_4bit_ex sidecar must have ex_bits > 0: " + ex_filename);
    }
}

}  // namespace

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

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

namespace gpu_mvr {

struct RefineTask {
    int local_idx;
    int doc_id;
    size_t out_offset;
    size_t num_tokens;
};

inline void compute_ip_ex_for_doc(
    const gpu_mvr_index& index,
    query_object* queries,
    const RefineTask& task,
    float* ip_ex_buf
) {
    const size_t doc_start = index.doc_ptrs_[task.doc_id];
    for (size_t j = 0; j < Q_DOCLEN; ++j) {
        for (size_t t = 0; t < task.num_tokens; ++t) {
            const size_t tid = doc_start + t;
            ip_ex_buf[(task.out_offset + t) * Q_DOCLEN + j] = ip_ex_bits(
                queries + j,
                &index.ex_code_[tid * PADDED_DIM * index.ex_bits / 8],
                index.ip_func_,
                PADDED_DIM
            );
        }
    }
}

inline void compute_ip_ex_for_doc_t_major(
    const gpu_mvr_index& index,
    query_object* queries,
    const RefineTask& task,
    float* ip_ex_buf
) {
    const size_t doc_start = index.doc_ptrs_[task.doc_id];
    for (size_t t = 0; t < task.num_tokens; ++t) {
        const size_t tid = doc_start + t;
        const char* ex_code = &index.ex_code_[tid * PADDED_DIM * index.ex_bits / 8];
        for (size_t j = 0; j < Q_DOCLEN; ++j) {
            ip_ex_buf[(task.out_offset + t) * Q_DOCLEN + j] = ip_ex_bits(
                queries + j,
                ex_code,
                index.ip_func_,
                PADDED_DIM
            );
        }
    }
}

inline void compute_ip_ex_for_doc_avx512(
    const gpu_mvr_index& index,
    const RefineTask& task,
    const float* queries_flat,
    float* ip_ex_buf
) {
    const size_t doc_start = index.doc_ptrs_[task.doc_id];
    const size_t ex_code_stride = PADDED_DIM * index.ex_bits / 8;
    alignas(64) float decoded[PADDED_DIM];

    for (size_t t = 0; t < task.num_tokens; ++t) {
        const size_t tid = doc_start + t;
        index.unpack_func_(
            reinterpret_cast<const uint8_t*>(&index.ex_code_[tid * ex_code_stride]),
            decoded,
            PADDED_DIM);
        if (t + 1 < task.num_tokens) {
            __builtin_prefetch(&index.ex_code_[(tid + 1) * ex_code_stride], 0, 3);
        }
        rabitqlib::gemv_batch8_avx512(
            queries_flat,
            decoded,
            &ip_ex_buf[(task.out_offset + t) * Q_DOCLEN],
            Q_DOCLEN,
            PADDED_DIM);
    }
}

inline void combine_refined_score_for_doc(
    const gpu_mvr_index& index,
    query_object* queries,
    const RefineTask& task,
    const float* one_bit_dists,
    const float* ip_ex_buf,
    std::pair<float, int>* refined_scores
) {
    const size_t doc_start = index.doc_ptrs_[task.doc_id];
    float doc_score = 0.0f;
    for (size_t j = 0; j < Q_DOCLEN; ++j) {
        float max_ts = -std::numeric_limits<float>::infinity();
        for (size_t t = 0; t < task.num_tokens; ++t) {
            const size_t tid = doc_start + t;
            const float combined = combine_dists(
                queries + j,
                one_bit_dists[(task.out_offset + t) * Q_DOCLEN + j],
                ip_ex_buf[(task.out_offset + t) * Q_DOCLEN + j],
                index.one_bit_factor_[tid],
                index.ex_factor_[tid],
                index.ex_bits
            );
            max_ts = std::max(max_ts, combined);
        }
        doc_score += max_ts;
    }
    refined_scores[task.local_idx] = {doc_score, task.doc_id};
}

inline void combine_refined_score_for_doc_t_major(
    const gpu_mvr_index& index,
    query_object* queries,
    const RefineTask& task,
    const float* one_bit_dists,
    const float* ip_ex_buf,
    std::pair<float, int>* refined_scores
) {
    const size_t doc_start = index.doc_ptrs_[task.doc_id];
    float max_scores[Q_DOCLEN];
    for (size_t j = 0; j < Q_DOCLEN; ++j) {
        max_scores[j] = -std::numeric_limits<float>::infinity();
    }

    for (size_t t = 0; t < task.num_tokens; ++t) {
        const size_t tid = doc_start + t;
        const float one_bit_factor = index.one_bit_factor_[tid];
        const float ex_factor = index.ex_factor_[tid];
        for (size_t j = 0; j < Q_DOCLEN; ++j) {
            const float combined = combine_dists(
                queries + j,
                one_bit_dists[(task.out_offset + t) * Q_DOCLEN + j],
                ip_ex_buf[(task.out_offset + t) * Q_DOCLEN + j],
                one_bit_factor,
                ex_factor,
                index.ex_bits
            );
            max_scores[j] = std::max(max_scores[j], combined);
        }
    }

    float doc_score = 0.0f;
    for (size_t j = 0; j < Q_DOCLEN; ++j) {
        doc_score += max_scores[j];
    }
    refined_scores[task.local_idx] = {doc_score, task.doc_id};
}

inline void combine_refined_score_for_doc_avx512(
    const gpu_mvr_index& index,
    const RefineTask& task,
    const float* query_bias,
    const float* one_bit_dists,
    const float* ip_ex_buf,
    std::pair<float, int>* refined_scores
) {
    const size_t doc_start = index.doc_ptrs_[task.doc_id];
    const float scale = static_cast<float>(1 << index.ex_bits);
    alignas(64) float max_ts[Q_DOCLEN];
    const __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    for (size_t j = 0; j < Q_DOCLEN; j += 16) {
        _mm512_store_ps(&max_ts[j], neg_inf);
    }

    for (size_t t = 0; t < task.num_tokens; ++t) {
        const size_t tid = doc_start + t;
        const __m512 v_tok_scale = _mm512_set1_ps(scale / index.one_bit_factor_[tid]);
        const __m512 v_tok_exf = _mm512_set1_ps(index.ex_factor_[tid]);
        const float* ob_base = &one_bit_dists[(task.out_offset + t) * Q_DOCLEN];
        const float* ex_base = &ip_ex_buf[(task.out_offset + t) * Q_DOCLEN];

        for (size_t j = 0; j < Q_DOCLEN; j += 16) {
            const __m512 ob = _mm512_loadu_ps(&ob_base[j]);
            const __m512 exd = _mm512_loadu_ps(&ex_base[j]);
            const __m512 qb = _mm512_load_ps(&query_bias[j]);
            const __m512 combined = _mm512_mul_ps(
                _mm512_add_ps(_mm512_fmadd_ps(v_tok_scale, ob, qb), exd),
                v_tok_exf);
            _mm512_store_ps(
                &max_ts[j],
                _mm512_max_ps(_mm512_load_ps(&max_ts[j]), combined));
        }
    }

    __m512 sum = _mm512_load_ps(&max_ts[0]);
    for (size_t j = 16; j < Q_DOCLEN; j += 16) {
        sum = _mm512_add_ps(sum, _mm512_load_ps(&max_ts[j]));
    }
    refined_scores[task.local_idx] = {_mm512_reduce_add_ps(sum), task.doc_id};
}

namespace v1 {

inline void compute_ip_ex_bits_cpu(
    const gpu_mvr_index& index,
    query_object* queries,
    const std::vector<RefineTask>& tasks,
    float* ip_ex_buf
) {
#pragma omp parallel for schedule(dynamic)
    for (size_t task_idx = 0; task_idx < tasks.size(); ++task_idx) {
        compute_ip_ex_for_doc(index, queries, tasks[task_idx], ip_ex_buf);
    }
}

inline void combine_refined_scores_cpu(
    const gpu_mvr_index& index,
    query_object* queries,
    const std::vector<RefineTask>& tasks,
    const float* one_bit_dists,
    const float* ip_ex_buf,
    std::pair<float, int>* refined_scores
) {
#pragma omp parallel for schedule(dynamic)
    for (size_t task_idx = 0; task_idx < tasks.size(); ++task_idx) {
        combine_refined_score_for_doc(
            index, queries, tasks[task_idx], one_bit_dists, ip_ex_buf, refined_scores);
    }
}

}  // namespace v1

namespace v2 {

struct TokenBalancedBuckets {
    std::vector<std::vector<int>> buckets;
    std::vector<size_t> thread_loads;
};

inline TokenBalancedBuckets build_token_balanced_buckets(
    const std::vector<RefineTask>& tasks,
    int thread_count
) {
    if (thread_count <= 0) thread_count = 1;
    thread_count = std::min<int>(thread_count, std::max<size_t>(1, tasks.size()));

    std::vector<int> order(tasks.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&tasks](int a, int b) {
        return tasks[a].num_tokens > tasks[b].num_tokens;
    });

    std::vector<std::vector<int>> buckets(thread_count);
    std::vector<size_t> thread_loads(thread_count, 0);
    for (int task_idx : order) {
        int target_thread = 0;
        for (int tid = 1; tid < thread_count; ++tid) {
            if (thread_loads[tid] < thread_loads[target_thread]) {
                target_thread = tid;
            }
        }
        buckets[target_thread].push_back(task_idx);
        thread_loads[target_thread] += tasks[task_idx].num_tokens;
    }
    return TokenBalancedBuckets{std::move(buckets), std::move(thread_loads)};
}

inline void compute_ip_ex_bits_cpu(
    const gpu_mvr_index& index,
    query_object* queries,
    const std::vector<RefineTask>& tasks,
    const float* queries_flat,
    float* ip_ex_buf
) {
    const int thread_count = std::min<int>(omp_get_max_threads(), std::max<size_t>(1, tasks.size()));
    const auto assignment = build_token_balanced_buckets(tasks, thread_count);
#pragma omp parallel num_threads(thread_count)
    {
        const int tid = omp_get_thread_num();
        const auto& bucket = assignment.buckets[tid];
        for (size_t bucket_pos = 0; bucket_pos < bucket.size(); ++bucket_pos) {
            const int task_idx = bucket[bucket_pos];
            compute_ip_ex_for_doc_avx512(index, tasks[task_idx], queries_flat, ip_ex_buf);
        }
    }
}

inline void combine_refined_scores_cpu(
    const gpu_mvr_index& index,
    query_object* queries,
    const std::vector<RefineTask>& tasks,
    const float* one_bit_dists,
    const float* ip_ex_buf,
    std::pair<float, int>* refined_scores
) {
    const int thread_count = std::min<int>(omp_get_max_threads(), std::max<size_t>(1, tasks.size()));
    const auto assignment = build_token_balanced_buckets(tasks, thread_count);
    alignas(64) float query_bias[Q_DOCLEN];
    const float scale = static_cast<float>(1 << index.ex_bits);
    for (size_t j = 0; j < Q_DOCLEN; ++j) {
        query_bias[j] = scale * queries[j].cb1_sumq - queries[j].cbex_sumq;
    }
#pragma omp parallel num_threads(thread_count)
    {
        const int tid = omp_get_thread_num();
        const auto& bucket = assignment.buckets[tid];
        for (size_t bucket_pos = 0; bucket_pos < bucket.size(); ++bucket_pos) {
            const int task_idx = bucket[bucket_pos];
            combine_refined_score_for_doc_avx512(
                index, tasks[task_idx], query_bias, one_bit_dists, ip_ex_buf, refined_scores);
        }
    }
}

}  // namespace v2
}  // namespace gpu_mvr

gpu_mvr_index::gpu_mvr_index(
        const std::string& filename,
        const std::vector<int>& doc_lens,
        const gpu_search_runtime_options& runtime_options) {
        nprobe = runtime_options.nprobe;
        k_rank_cluster = runtime_options.k_rank_cluster;
        k_rank_all_tokens = runtime_options.k_rank_all_tokens;
        itopk_size = runtime_options.itopk_size;
        overlap_chunks = runtime_options.overlap_chunks;
        gpu_mvr::StartupProfile startup("index_ctor");
        const auto resolved_paths = gpu_index_layout::resolve_index_paths(filename);
        startup.mark("resolve_index_paths");

        std::ifstream inf(resolved_paths.quantized_data_path, std::ios::binary);
        const auto header =
            mvr_index_file_format::read_header(inf, resolved_paths.quantized_data_path);
        n = header.n;
        d = header.d;
        n_clusters = header.n_clusters;
        ex_bits = header.ex_bits;

        if (header.padded_dim != PADDED_DIM) {
            inf.close();
            throw std::runtime_error(
                "Index file padded_dim=" + std::to_string(header.padded_dim) +
                " does not match compiled PADDED_DIM=" + std::to_string(PADDED_DIM) +
                ". Please recompile with matching PADDED_DIM in gpu_config.cuh"
            );
        }

        rotator_ = mvr_index_file_format::load_rotator(inf, header, filename);
        startup.mark("read_header_and_rotator");

        one_bit_code_.resize(n * PADDED_DIM / 8);
        one_bit_factor_.resize(n);

        inf.read(one_bit_code_.data(), one_bit_code_.size());
        if (!inf) {
            throw std::runtime_error(
                "Failed to read one-bit codes from index file: " +
                resolved_paths.quantized_data_path
            );
        }

        inf.read((char*)one_bit_factor_.data(), n * sizeof(float));
        if (!inf) {
            throw std::runtime_error(
                "Failed to read one-bit scaling factors from index file: " +
                resolved_paths.quantized_data_path
            );
        }
        const bool has_doc4_ex =
            std::filesystem::exists(resolved_paths.doc_4bit_ex_path);
        if (has_doc4_ex) {
            std::ifstream doc4ex(resolved_paths.doc_4bit_ex_path, std::ios::binary);
            const auto doc4ex_header =
                mvr_index_file_format::read_header(doc4ex, resolved_paths.doc_4bit_ex_path);
            validate_doc4ex_header_compatibility(
                header,
                doc4ex_header,
                resolved_paths.quantized_data_path,
                resolved_paths.doc_4bit_ex_path);
            auto* doc4ex_rotator = mvr_index_file_format::load_rotator(
                doc4ex, doc4ex_header, resolved_paths.doc_4bit_ex_path);
            delete doc4ex_rotator;

            ex_bits = doc4ex_header.ex_bits;
            ex_code_.resize(n * PADDED_DIM * ex_bits / 8);
            ex_factor_.resize(n);
            startup.note("quantized_payload_mode", "doc_4bit_ex_sidecar");
            doc4ex.read(ex_code_.data(), ex_code_.size());
            doc4ex.read((char*)ex_factor_.data(), n * sizeof(float));
            if (!doc4ex) {
                throw std::runtime_error(
                    "Failed to read doc_4bit_ex payload from index file: " +
                    resolved_paths.doc_4bit_ex_path);
            }
        } else {
            ex_code_.resize(n * PADDED_DIM * ex_bits / 8);
            ex_factor_.resize(n);
            const size_t ex_code_stride = PADDED_DIM * ex_bits / 8;
            const size_t full_code_stride = PADDED_DIM * (1 + ex_bits) / 8;
            startup.note("quantized_payload_mode", "doc_4bit_to_ex_decode");
            inf.read((char*)ex_factor_.data(), n * sizeof(float));
            if (!inf) {
                throw std::runtime_error(
                    "Failed to read scaling factors from index file: " +
                    resolved_paths.quantized_data_path
                );
            }

            std::ifstream doc4(resolved_paths.doc_4bit_path, std::ios::binary);
            const auto doc4_header =
                mvr_index_file_format::read_header(doc4, resolved_paths.doc_4bit_path);
            mvr_index_file_format::validate_matching_header(
                header,
                doc4_header,
                resolved_paths.quantized_data_path,
                resolved_paths.doc_4bit_path);
            auto* doc4_rotator =
                mvr_index_file_format::load_rotator(doc4, doc4_header, resolved_paths.doc_4bit_path);
            delete doc4_rotator;

            auto full_unpack = select_excode_unpackfunc(1 + ex_bits);
            const uint8_t ex_mask = static_cast<uint8_t>((1u << ex_bits) - 1u);
            const size_t batch_vectors = 8192;
            std::vector<char> full_batch(batch_vectors * full_code_stride);
            std::vector<float> unpacked(PADDED_DIM);
            std::vector<uint8_t> raw_ex(PADDED_DIM);

            for (size_t start = 0; start < n; start += batch_vectors) {
                const size_t batch_count = std::min(batch_vectors, n - start);
                const size_t batch_bytes = batch_count * full_code_stride;
                doc4.read(full_batch.data(), batch_bytes);
                if (!doc4) {
                    throw std::runtime_error(
                        "Failed to read doc_4bit payload from index file: " +
                        resolved_paths.doc_4bit_path
                    );
                }

                for (size_t i = 0; i < batch_count; ++i) {
                    full_unpack(
                        reinterpret_cast<const uint8_t*>(full_batch.data() + i * full_code_stride),
                        unpacked.data(),
                        PADDED_DIM);
                    for (size_t dim_idx = 0; dim_idx < PADDED_DIM; ++dim_idx) {
                        raw_ex[dim_idx] =
                            static_cast<uint8_t>(unpacked[dim_idx]) & ex_mask;
                    }
                    quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
                        raw_ex.data(),
                        reinterpret_cast<uint8_t*>(ex_code_.data() + (start + i) * ex_code_stride),
                        PADDED_DIM,
                        ex_bits);
                }
            }
        }
        startup.mark("read_quantized_payload");
        inf.close();
        startup.mark("read_scaling_factors");

        ip_func_ = select_excode_ipfunc(ex_bits);
        unpack_func_ = select_excode_unpackfunc(ex_bits);

        ivf = new IVF_PG(n_clusters, d, PGType::CAGRA);
        ivf->load(resolved_paths.ivf_path, resolved_paths.centroids_path);
        max_cluster_size = ivf->max_cluster_size();
        startup.mark("load_ivf");

        set_doc_mapping(doc_lens);
        startup.mark("set_doc_mapping");

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
        startup.mark("upload_persistent_data");

        // Allocate workspace
        allocate_workspace();
        startup.mark("allocate_workspace");
    }

void gpu_mvr_index::set_doc_mapping(const std::vector<int>& doc_lens) {
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

void gpu_mvr_index::allocate_workspace() {
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
        CUDA_CHECK(cudaMalloc(&ws_.d_stage2_doc_query_max, ws_.max_stage2_candidates * Q_DOCLEN * sizeof(float)));
#endif

        // Extract workspace (used by both paths)
        CUDA_CHECK(cudaMalloc(&ws_.d_selected_indices, ws_.max_stage2_k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_out_offsets, (ws_.max_stage2_k + 1) * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_out_one_bit_dists, ws_.max_stage2_k_tokens * Q_DOCLEN * sizeof(float)));

#if GPU_MVR_BINARY_IP_IMPL == 2
        // LUT workspace for binary IP
        CUDA_CHECK(cudaMalloc(&ws_.d_lut, LUT_TOTAL_FLOATS * sizeof(float)));
#endif
        gpu_mvr::configure_binary_ip_kernels();

        // CAGRA batched search workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_cagra_dists, Q_DOCLEN * nprobe * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_cagra_labels, Q_DOCLEN * nprobe * sizeof(pg_label_t)));

        // Pinned host memory
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_cb1_sumq, Q_DOCLEN * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_dists, ws_.max_stage2_k_tokens * Q_DOCLEN * sizeof(float)));
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
        CUDA_CHECK(cudaMalloc(&ws_.d_stage2_doc_query_max,
                              ws_.max_stage2_candidates * Q_DOCLEN * sizeof(float)));

        // Pinned host memory for D2H of doc scores
        CUDA_CHECK(cudaMallocHost(&ws_.h_mapped_doc_scores,
                                  ws_.max_stage2_candidates * sizeof(float)));

        ws_.v_pst_candidate_offsets.resize(ws_.max_stage2_candidates + 1);

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
#if GPU_MVR_STAGE1_SPARSE_CLEANUP == 1
        CUDA_CHECK(cudaEventCreateWithFlags(&ws_.event_stage1_sum_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ws_.event_stage1_cleanup_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(ws_.event_stage1_cleanup_done, ws_.stream_h2d));
#endif

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
        ws_.running_indices.resize(ws_.max_stage2_candidates);
        ws_.h_sel_indices.resize(ws_.max_stage2_k);
        ws_.h_out_offsets.resize(ws_.max_stage2_k + 1);
        ws_.ip_ex_buf.resize(ws_.max_stage2_k * (size_t)max_doc_len * Q_DOCLEN);
        ws_.refined_scores.resize(ws_.max_stage2_k);
    }

size_t gpu_mvr_index::doc_len(size_t doc_id) const {
        return doc_ptrs_[doc_id + 1] - doc_ptrs_[doc_id];
    }

    // ======================== SEARCH PIPELINE ========================

std::vector<size_t> gpu_mvr_index::search(const float* queries, size_t k) {
        return search_impl<false>(queries, k);
    }

std::vector<size_t> gpu_mvr_index::search_profiled(const float* queries, size_t k) {
        return search_impl<true>(queries, k);
    }

template <bool kProfile>
std::vector<size_t> gpu_mvr_index::search_impl(const float* queries, size_t k) {
        GPU_MVR_NVTX_SCOPE("gpu_index.search", 0xFF2A9D8F);
#ifdef GPU_MVR_PROFILE
        auto search_start = std::chrono::high_resolution_clock::now();
        if constexpr (kProfile) {
            ws_.xfer_count = 0;  // reset transfer profiling counters
        }
#endif

#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.event_start, ws_.stream_compute));
        }
#endif

        // Step 1: Rotate queries on CPU into pinned memory (async-ready)
        {
            GPU_MVR_NVTX_SCOPE("search.rotate_queries_cpu", 0xFF457B9D);
            for (size_t i = 0; i < Q_DOCLEN; ++i) {
                rotator_->rotate(&queries[i * d], &ws_.h_pinned_queries[i * PADDED_DIM]);
            }
        }

        // Build query objects on CPU
        std::vector<query_object> query_objs(Q_DOCLEN);
        {
            GPU_MVR_NVTX_SCOPE("search.build_query_objects", 0xFF577590);
            for (size_t i = 0; i < Q_DOCLEN; ++i) {
                query_objs[i] = query_object(&ws_.h_pinned_queries[i * PADDED_DIM], PADDED_DIM, ex_bits);
                ws_.h_pinned_cb1_sumq[i] = query_objs[i].cb1_sumq;
            }
        }

        // Upload from pinned memory using async stream
        {
            GPU_MVR_NVTX_SCOPE("search.h2d_queries", 0xFF4D908E);
            if constexpr (kProfile) {
                XFER_RECORD_BEGIN(ws_.stream_h2d);
            }
            CUDA_CHECK(cudaMemcpyAsync(ws_.d_queries, ws_.h_pinned_queries,
                                       Q_DOCLEN * PADDED_DIM * sizeof(float),
                                       cudaMemcpyHostToDevice, ws_.stream_h2d));
            CUDA_CHECK(cudaMemcpyAsync(ws_.d_cb1_sumq, ws_.h_pinned_cb1_sumq,
                                       Q_DOCLEN * sizeof(float),
                                       cudaMemcpyHostToDevice, ws_.stream_h2d));
            if constexpr (kProfile) {
                XFER_RECORD_END(ws_.stream_h2d, Q_DOCLEN * PADDED_DIM * sizeof(float) + Q_DOCLEN * sizeof(float), true);
            }
        }

        // Wait for H2D transfer before compute
        CUDA_CHECK(cudaEventRecord(ws_.event_h2d_done, ws_.stream_h2d));
        CUDA_CHECK(cudaStreamWaitEvent(ws_.stream_compute, ws_.event_h2d_done));

        {
            GPU_MVR_NVTX_SCOPE("stage0.precompute_lut", 0xFF43AA8B);
            gpu_mvr::precompute_binary_ip_state(
                ws_.d_queries,
#if GPU_MVR_BINARY_IP_IMPL == 2
                ws_.d_lut,
#else
                nullptr,
#endif
                ws_.stream_compute);
            CUDA_CHECK(cudaGetLastError());
        }

#ifdef GPU_MVR_PROFILE
#if GPU_MVR_STAGE1_SPARSE_CLEANUP == 1
        // Use the latest cleanup completion as the logical start boundary for Stage 1 profiling.
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventSynchronize(ws_.event_stage1_cleanup_done));
        }
#endif
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.event_stage1_start, ws_.stream_compute));
        }
#endif

        // Stage 1: GPU-only aggregation and top-k
        int actual_k_stage1 = 0;
        rank_cluster_dists_gpu_impl<kProfile>(query_objs.data(), nprobe, k_rank_cluster,
                              actual_k_stage1, ws_.stream_compute);

#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.event_stage1_end, ws_.stream_compute));
            CUDA_CHECK(cudaEventRecord(ws_.event_stage2_start, ws_.stream_compute));
        }
#endif

        std::vector<size_t> result;

#ifdef GPU_MVR_OVERLAP_STAGE23
        rank_stage23_persistent_impl<kProfile>(actual_k_stage1, k, k_rank_all_tokens, query_objs.data(), result);

#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.event_stage2_end, ws_.stream_compute));
            CUDA_CHECK(cudaEventRecord(ws_.event_end, ws_.stream_compute));
            CUDA_CHECK(cudaEventSynchronize(ws_.event_end));
        }

        if constexpr (kProfile) {
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
            std::cout << "[PROFILE]   1. Coarse search           : " << s1_cagra << " ms\n";
            std::cout << "[PROFILE]   2. GPU IVF expansion       : " << s1_expansion << " ms\n";
            std::cout << "[PROFILE]   3. Binary IP kernel        : " << s1_binary_ip << " ms\n";
            std::cout << "[PROFILE]   4. Aggregation + tracking  : " << s1_atomic_agg << " ms\n";
            std::cout << "[PROFILE]   5. Sum doc scores (sparse) : " << s1_sum_scores << " ms\n";
            std::cout << "[PROFILE]   6. Top-k sort (sparse)     : " << s1_topk_sort << " ms\n";
            std::cout << "[PROFILE]   7. D2D copy top-k doc IDs  : " << s1_d2d << " ms\n";
            std::cout << "[PROFILE]   8. Memset (overlapped 1-3) : " << s1_memset << " ms (not in critical path)\n";
            float s1_sum = s1_cagra + s1_expansion + s1_binary_ip + s1_atomic_agg + s1_sum_scores + s1_topk_sort + s1_d2d;
            std::cout << "[PROFILE]   Sum accounted              : " << s1_sum << " ms\n";
            std::cout << "[PROFILE] Total search time           : " << total_time << " ms\n";

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

        rank_all_tokens_1bit_gpu_impl<kProfile>(actual_k_stage1, k_rank_all_tokens,
                                                stage2_doc_ids, stage2_one_bit_dists,
                                                ws_.stream_compute);

#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.event_stage2_end, ws_.stream_compute));
            CUDA_CHECK(cudaEventSynchronize(ws_.event_stage2_end));
        }
#endif

        // Stage 3: CPU ex-bits refinement (sequential, no overlap)
#ifdef GPU_MVR_PROFILE
        auto stage3_wall_start = std::chrono::high_resolution_clock::now();
#endif

        rank_all_tokens_exbits_cpu(query_objs.data(),
                                   stage2_doc_ids, stage2_one_bit_dists,
                                   k, result);

#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
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
            std::cout << "[PROFILE]   1. Coarse search           : " << s1_cagra << " ms\n";
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

void gpu_mvr_index::rank_cluster_dists_gpu(
        query_object* h_query_objs,
        size_t nprobe, size_t k,
        int& actual_k_out,
        cudaStream_t stream
    ) {
        rank_cluster_dists_gpu_impl<false>(h_query_objs, nprobe, k, actual_k_out, stream);
    }

template <bool kProfile>
void gpu_mvr_index::rank_cluster_dists_gpu_impl(
        query_object* h_query_objs,
        size_t nprobe, size_t k,
        int& actual_k_out,
        cudaStream_t stream
    ) {
        GPU_MVR_NVTX_SCOPE("stage1.rank_cluster_dists_gpu", 0xFFF4A261);
        int total_pairs = 0;
        int h_num_touched = 0;
        int max_embs_per_query = ws_.max_embs_per_query_bound;
        int thread_count = 256;

        size_t matrix_size = (size_t)num_docs * Q_DOCLEN;
#if GPU_MVR_STAGE1_SPARSE_CLEANUP == 0
        // Launch memset on d2h stream (overlaps with CAGRA + expansion + binary IP)
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_memset_start, ws_.stream_d2h));
        }
#endif
        CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_query_max, 0, matrix_size * sizeof(float), ws_.stream_d2h));
        CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_touched, 0, ws_.estimated_num_docs * sizeof(int), ws_.stream_d2h));
        CUDA_CHECK(cudaMemsetAsync(ws_.d_num_unique_docs, 0, sizeof(int), ws_.stream_d2h));
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_memset_end, ws_.stream_d2h));
        }
#endif
        CUDA_CHECK(cudaEventRecord(ws_.event_h2d_done, ws_.stream_d2h));  // reuse for memset sync
#endif

#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_cagra_start, stream));
        }
#endif

        // 1. Batched CAGRA search: one call for all Q_DOCLEN queries on GPU
        //    ws_.d_queries is already on GPU from the caller.
        {
            GPU_MVR_NVTX_SCOPE("stage1.cagra_search", 0xFFE76F51);
            ivf->search_batch_gpu(
                ws_.d_queries,
                Q_DOCLEN,
                nprobe,
                ws_.d_cagra_dists,
                ws_.d_cagra_labels,
                stream,
                static_cast<size_t>(itopk_size));
        }

#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_cagra_end, stream));
            CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_start, stream));
        }
#endif

        // GPU-based cluster expansion (replaces CPU expansion and transfers)
        // Step 1: Compute per-query expansion sizes on GPU
        {
            GPU_MVR_NVTX_SCOPE("stage1.expand_candidates", 0xFFE9C46A);
            compute_query_expansion_sizes_kernel<<<(Q_DOCLEN + 255) / 256, 256, 0, stream>>>(
                ws_.d_cagra_labels,
                d_cluster_pos_,
                ws_.d_pair_offsets + 1,
                nprobe,
                ivf->n_clusters,
                Q_DOCLEN
            );

            CUDA_CHECK(cudaMemsetAsync(ws_.d_pair_offsets, 0, sizeof(int), stream));

            thrust::device_ptr<int> offsets_ptr(ws_.d_pair_offsets + 1);
            thrust::inclusive_scan(thrust::cuda::par.on(stream),
                                   offsets_ptr, offsets_ptr + Q_DOCLEN, offsets_ptr);

            // Step 4: Get total pairs count
            if constexpr (kProfile) {
                XFER_RECORD_BEGIN(stream);
            }
            CUDA_CHECK(cudaMemcpyAsync(&total_pairs, ws_.d_pair_offsets + Q_DOCLEN,
                                       sizeof(int), cudaMemcpyDeviceToHost, stream));
            if constexpr (kProfile) {
                XFER_RECORD_END(stream, sizeof(int), false);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            if (total_pairs == 0) {
                actual_k_out = 0;
                return;
            }

            expand_cluster_ids_kernel<<<Q_DOCLEN, 256, 0, stream>>>(
                ws_.d_cagra_labels,
                d_inv_list_,
                d_cluster_pos_,
                ws_.d_pair_offsets,
                ws_.d_emb_ids,
                nprobe,
                ivf->n_clusters,
                Q_DOCLEN
            );

#ifdef GPU_MVR_PROFILE
            if constexpr (kProfile) {
                CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_end, stream));
            }
#endif
            // Note: ws_.d_emb_ids and ws_.d_pair_offsets are now ready for Stage 1 kernel

            // 2. Compute 1-bit distances using optimized shared-memory kernel
            // Use precomputed upper bound for max_embs_per_query
            // (eliminates thrust::device_vector alloc + adjacent_difference + reduce per search)
            int threads_per_block = 256;
#ifdef GPU_MVR_PROFILE
            if constexpr (kProfile) {
                CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_start, stream));
            }
#endif
            {
                GPU_MVR_NVTX_SCOPE("stage1.binary_ip", 0xFFE76F51);
                gpu_mvr::launch_stage1_binary_ip(
                    ws_.d_queries,
#if GPU_MVR_BINARY_IP_IMPL == 2
                    ws_.d_lut,
#else
                    nullptr,
#endif
                    d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
                    ws_.d_emb_ids, ws_.d_pair_offsets, ws_.d_emb_dists,
                    max_embs_per_query, threads_per_block, stream
                );
                CUDA_CHECK(cudaGetLastError());
            }
#ifdef GPU_MVR_PROFILE
            if constexpr (kProfile) {
                CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_end, stream));
            }
#endif

            // === Sparse aggregation: only process documents actually hit ===
            // Wait for cleanup/reset to complete before aggregation.
#if GPU_MVR_STAGE1_SPARSE_CLEANUP == 0
            // Full reset is overlapped on stream_d2h.
            CUDA_CHECK(cudaStreamWaitEvent(stream, ws_.event_h2d_done));
#else
            // Sparse cleanup runs after the previous search on stream_h2d.
            CUDA_CHECK(cudaStreamWaitEvent(stream, ws_.event_stage1_cleanup_done));
#ifdef GPU_MVR_PROFILE
            CUDA_CHECK(cudaEventRecord(ws_.s1_memset_start, ws_.stream_h2d));
            CUDA_CHECK(cudaEventRecord(ws_.s1_memset_end, ws_.stream_h2d));
#endif
#endif

            // 4. Atomic aggregation with touched-doc tracking
            int block_count = (total_pairs + thread_count - 1) / thread_count;

#ifdef GPU_MVR_PROFILE
            if constexpr (kProfile) {
                CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_start, stream));
            }
#endif
            {
                GPU_MVR_NVTX_SCOPE("stage1.aggregate_scores", 0xFFF4A261);
                aggregate_stage1_tracked_kernel<<<block_count, thread_count, 0, stream>>>(
                    ws_.d_emb_ids, ws_.d_emb_dists, ws_.d_pair_offsets, d_doc_ids_,
                    ws_.d_doc_query_max,
                    ws_.d_doc_touched, ws_.d_unique_doc_ids, ws_.d_num_unique_docs,
                    num_docs, total_pairs, max_embs_per_query
                );
                CUDA_CHECK(cudaGetLastError());
            }
#ifdef GPU_MVR_PROFILE
            if constexpr (kProfile) {
                CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_end, stream));
            }
#endif

            // 5. Get number of touched docs (single int D2H)
            if constexpr (kProfile) {
                XFER_RECORD_BEGIN(stream);
            }
            CUDA_CHECK(cudaMemcpyAsync(&h_num_touched, ws_.d_num_unique_docs,
                                       sizeof(int), cudaMemcpyDeviceToHost, stream));
            if constexpr (kProfile) {
                XFER_RECORD_END(stream, sizeof(int), false);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            if (h_num_touched == 0) {
                actual_k_out = 0;
                return;
            }
            actual_k_out = std::min((int)k, h_num_touched);

            // 6. Sum across queries only for touched docs (sparse)
            int sparse_blocks = (h_num_touched + thread_count - 1) / thread_count;
#ifdef GPU_MVR_PROFILE
            if constexpr (kProfile) {
                CUDA_CHECK(cudaEventRecord(ws_.s1_sum_scores_start, stream));
            }
#endif
            sum_doc_scores_sparse_kernel<<<sparse_blocks, thread_count, 0, stream>>>(
                ws_.d_doc_query_max,
                ws_.d_unique_doc_ids,
                ws_.d_stage1_doc_scores,
                ws_.d_topk_doc_ids,
                h_num_touched, num_docs
            );
        }
        CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_sum_scores_end, stream));
        }
#endif

#if GPU_MVR_STAGE1_SPARSE_CLEANUP == 1
        CUDA_CHECK(cudaEventRecord(ws_.event_stage1_sum_done, stream));

        CUDA_CHECK(cudaStreamWaitEvent(ws_.stream_h2d, ws_.event_stage1_sum_done));
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_memset_start, ws_.stream_h2d));
        }
#endif
        cleanup_touched_docs_kernel<<<sparse_blocks, thread_count, 0, ws_.stream_h2d>>>(
            ws_.d_unique_doc_ids,
            ws_.d_doc_query_max,
            ws_.d_doc_touched,
            h_num_touched,
            num_docs
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemsetAsync(ws_.d_num_unique_docs, 0, sizeof(int), ws_.stream_h2d));
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_memset_end, ws_.stream_h2d));
        }
#endif
        CUDA_CHECK(cudaEventRecord(ws_.event_stage1_cleanup_done, ws_.stream_h2d));
#endif

        // 7. Top-k: sort only touched docs by score descending
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_topk_sort_start, stream));
        }
#endif
        cub::DeviceRadixSort::SortPairsDescending(
            ws_.d_cub_temp_storage, ws_.cub_temp_storage_bytes,
            ws_.d_stage1_doc_scores, ws_.d_topk_scores,
            ws_.d_topk_doc_ids, ws_.d_sorted_doc_ids,
            h_num_touched, 0, 32, stream
        );
        CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_topk_sort_end, stream));
        }
#endif

        // Top-k doc IDs are in d_sorted_doc_ids[0..actual_k_out-1]
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_d2d_start, stream));
        }
#endif
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_topk_doc_ids, ws_.d_sorted_doc_ids,
                                   actual_k_out * sizeof(int), cudaMemcpyDeviceToDevice, stream));
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_d2d_end, stream));
        }
#endif
        // Top-k doc IDs now in ws_.d_topk_doc_ids[0..actual_k_out-1]
        // These remain on GPU for Stage 2 consumption

    }

    // ======================== STAGE 2: GPU ========================

void gpu_mvr_index::rank_all_tokens_1bit_gpu(
        int num_candidates,  // from Stage 1
        size_t k,
        std::vector<size_t>& output_ids,
        std::vector<float>& one_bit_dists,
        cudaStream_t stream
    ) {
        rank_all_tokens_1bit_gpu_impl<false>(num_candidates, k, output_ids, one_bit_dists, stream);
    }

template <bool kProfile>
void gpu_mvr_index::rank_all_tokens_1bit_gpu_impl(
        int num_candidates,  // from Stage 1
        size_t k,
        std::vector<size_t>& output_ids,
        std::vector<float>& one_bit_dists,
        cudaStream_t stream
    ) {
        GPU_MVR_NVTX_SCOPE("stage2.rank_all_tokens_1bit_gpu", 0xFF264653);
        if (num_candidates == 0) return;

#ifdef GPU_MVR_PROFILE
        float s2_gather_ms = 0, s2_prefix_ms = 0, s2_tokenids_ms = 0, s2_binaryip_ms = 0;
        float s2_docscore_ms = 0, s2_topk_ms = 0, s2_d2h_offsets_ms = 0, s2_extract_ms = 0, s2_d2h_dists_ms = 0;
#endif

        // === NEW: Compute offsets on GPU ===
        // 1. Gather document lengths
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s2_gather_start, stream));
        }
#endif
        int threads = 256;
        int blocks = (num_candidates + threads - 1) / threads;

        gather_doc_lengths_kernel<<<blocks, threads, 0, stream>>>(
            ws_.d_topk_doc_ids, d_doc_ptrs_,
            ws_.d_pair_doc_ids,  // reuse as doc_lengths buffer
            num_candidates
        );
        CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s2_gather_end, stream));
        }
#endif

        // 2. Compute prefix sum to get candidate_offsets (int lengths → size_t offsets)
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s2_prefix_start, stream));
        }
#endif
        // size_t zero_val = 0;
        // CUDA_CHECK(cudaMemcpyAsync(ws_.d_candidate_offsets, &zero_val, sizeof(size_t),
        //                            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(ws_.d_candidate_offsets, 0, sizeof(size_t), stream));
        thrust::device_ptr<int> len_ptr(ws_.d_pair_doc_ids);
        thrust::device_ptr<size_t> off_ptr(ws_.d_candidate_offsets + 1);
        // Cast int lengths to size_t before accumulation to avoid type mismatch
        auto len_cast = thrust::make_transform_iterator(len_ptr, cast_int_size_t{});
        thrust::inclusive_scan(thrust::cuda::par.on(stream),
                              len_cast, len_cast + num_candidates,
                              off_ptr);

        // Get total tokens
        size_t total_tokens;
        if constexpr (kProfile) {
            XFER_RECORD_BEGIN(stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(&total_tokens, ws_.d_candidate_offsets + num_candidates,
                                   sizeof(size_t), cudaMemcpyDeviceToHost, stream));
        if constexpr (kProfile) {
            XFER_RECORD_END(stream, sizeof(size_t), false);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s2_prefix_end, stream));
        }
#endif

        // 3. Build token IDs on GPU
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s2_tokenids_start, stream));
        }
#endif
        gather_token_ids_kernel<<<num_candidates, 256, 0, stream>>>(
            ws_.d_topk_doc_ids,
            d_doc_ptrs_,
            ws_.d_candidate_offsets,
            ws_.d_token_ids,
            num_candidates
        );
        CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s2_tokenids_end, stream));
        }
#endif

        // 4. Compute all (query, token) 1-bit distances — multi-query fused kernel
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s2_binaryip_start, stream));
        }
#endif
        int threads_per_block = 256;

        gpu_mvr::launch_stage2_binary_ip(
            ws_.d_queries,
#if GPU_MVR_BINARY_IP_IMPL == 2
            ws_.d_lut,
#else
            nullptr,
#endif
            d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
            ws_.d_token_ids, ws_.d_token_dists,
            total_tokens, total_tokens, threads_per_block, stream
        );
        CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s2_binaryip_end, stream));
        }
#endif

        // 5. Compute doc scores with shared-memory reduction
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s2_docscore_start, stream));
        }
#endif
        int score_threads = 128;
        while (score_threads < Q_DOCLEN && score_threads < 256) score_threads *= 2;
        score_threads = std::min(score_threads, 256);

        gpu_mvr::launch_doc_score(
            ws_.d_token_dists, ws_.d_candidate_offsets, ws_.d_doc_scores,
            ws_.d_stage2_doc_query_max, total_tokens, num_candidates,
            score_threads, stream
        );
        CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s2_docscore_end, stream));
        }
#endif

        // 6. Top-k via CUB sort (descending)
#ifdef GPU_MVR_PROFILE
        cudaEvent_t s2_topk_start{}, s2_topk_end{};
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventCreate(&s2_topk_start));
            CUDA_CHECK(cudaEventCreate(&s2_topk_end));
            CUDA_CHECK(cudaEventRecord(s2_topk_start, stream));
        }
#endif
        thrust::device_ptr<int> indices_ptr(ws_.d_selected_indices);
        thrust::sequence(thrust::cuda::par.on(stream),
                        indices_ptr, indices_ptr + num_candidates);

        cub::DeviceRadixSort::SortPairsDescending(
            ws_.d_cub_temp_storage, ws_.cub_temp_storage_bytes,
            ws_.d_doc_scores, ws_.d_topk_scores,  // reuse topk buffers
            ws_.d_selected_indices, ws_.d_topk_indices,
            num_candidates, 0, 32, stream
        );
        CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(s2_topk_end, stream));
        }
#endif

        size_t actual_k = std::min(k, (size_t)num_candidates);

        // 7. Build output offsets for selected docs (using CPU for small arrays)
        // Copy top-k indices and candidate offsets together, sync once
#ifdef GPU_MVR_PROFILE
        cudaEvent_t s2_d2h_offsets_start{}, s2_d2h_offsets_end{};
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventCreate(&s2_d2h_offsets_start));
            CUDA_CHECK(cudaEventCreate(&s2_d2h_offsets_end));
            CUDA_CHECK(cudaEventRecord(s2_d2h_offsets_start, stream));
        }
#endif
        std::vector<int> h_top_k_indices(actual_k);
        std::vector<size_t> candidate_offsets_cpu(num_candidates + 1);
        if constexpr (kProfile) {
            XFER_RECORD_BEGIN(stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(h_top_k_indices.data(), ws_.d_topk_indices,
                                   actual_k * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(candidate_offsets_cpu.data(), ws_.d_candidate_offsets,
                             (num_candidates + 1) * sizeof(size_t),
                             cudaMemcpyDeviceToHost, stream));
        if constexpr (kProfile) {
            XFER_RECORD_END(stream, actual_k * sizeof(int) + (num_candidates + 1) * sizeof(size_t), false);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(s2_d2h_offsets_end, stream));
        }
#endif

        std::vector<size_t> out_offsets(actual_k + 1, 0);
        for (size_t i = 0; i < actual_k; ++i) {
            size_t cand_idx = h_top_k_indices[i];
            out_offsets[i + 1] = out_offsets[i] +
                (candidate_offsets_cpu[cand_idx + 1] - candidate_offsets_cpu[cand_idx]);
        }
        size_t total_selected_tokens = out_offsets[actual_k];

        // Upload selection metadata to GPU
        if constexpr (kProfile) {
            XFER_RECORD_BEGIN(stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_selected_indices, h_top_k_indices.data(),
                                   actual_k * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_out_offsets, out_offsets.data(),
                                   (actual_k + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream));
        if constexpr (kProfile) {
            XFER_RECORD_END(stream, actual_k * sizeof(int) + (actual_k + 1) * sizeof(size_t), true);
        }

        // 8. Extract on GPU
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s2_extract_start, stream));
        }
#endif
        extract_one_bit_dists_kernel<<<actual_k, 256, 0, stream>>>(
            ws_.d_token_dists, ws_.d_candidate_offsets,
            ws_.d_selected_indices, ws_.d_out_one_bit_dists,
            ws_.d_out_offsets, total_tokens, actual_k
        );
        CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s2_extract_end, stream));
        }
#endif

        // 9. Copy only selected dists to pinned host memory
#ifdef GPU_MVR_PROFILE
        cudaEvent_t s2_d2h_dists_start{}, s2_d2h_dists_end{};
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventCreate(&s2_d2h_dists_start));
            CUDA_CHECK(cudaEventCreate(&s2_d2h_dists_end));
            CUDA_CHECK(cudaEventRecord(s2_d2h_dists_start, stream));
        }
#endif
        size_t copy_size = total_selected_tokens * Q_DOCLEN;
        if constexpr (kProfile) {
            XFER_RECORD_BEGIN(stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_dists, ws_.d_out_one_bit_dists,
                                   copy_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

        // Also get the top-k doc IDs (need to gather from original candidates)
        std::vector<int> h_candidate_doc_ids(num_candidates);
        CUDA_CHECK(cudaMemcpyAsync(h_candidate_doc_ids.data(), ws_.d_topk_doc_ids,
                             num_candidates * sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
        if constexpr (kProfile) {
            XFER_RECORD_END(stream, copy_size * sizeof(float) + num_candidates * sizeof(int), false);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
#ifdef GPU_MVR_PROFILE
        std::chrono::high_resolution_clock::time_point build_output_begin{};
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(s2_d2h_dists_end, stream));
            build_output_begin = std::chrono::high_resolution_clock::now();
        }
#endif

        // Build output
        output_ids.resize(actual_k);
        // one_bit_dists.resize(copy_size);

        for (size_t i = 0; i < actual_k; ++i) {
            output_ids[i] = h_candidate_doc_ids[h_top_k_indices[i]];
        }
        // memcpy(one_bit_dists.data(), ws_.h_pinned_dists, copy_size * sizeof(float));

#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            auto build_output_end = std::chrono::high_resolution_clock::now();
            float build_output_time = std::chrono::duration<float, std::milli>(
                build_output_end - build_output_begin).count();

            CUDA_CHECK(cudaEventElapsedTime(&s2_gather_ms, ws_.s2_gather_start, ws_.s2_gather_end));
            CUDA_CHECK(cudaEventElapsedTime(&s2_prefix_ms, ws_.s2_prefix_start, ws_.s2_prefix_end));
            CUDA_CHECK(cudaEventElapsedTime(&s2_tokenids_ms, ws_.s2_tokenids_start, ws_.s2_tokenids_end));
            CUDA_CHECK(cudaEventElapsedTime(&s2_binaryip_ms, ws_.s2_binaryip_start, ws_.s2_binaryip_end));
            CUDA_CHECK(cudaEventElapsedTime(&s2_docscore_ms, ws_.s2_docscore_start, ws_.s2_docscore_end));
            CUDA_CHECK(cudaEventElapsedTime(&s2_topk_ms, s2_topk_start, s2_topk_end));
            CUDA_CHECK(cudaEventElapsedTime(&s2_d2h_offsets_ms, s2_d2h_offsets_start, s2_d2h_offsets_end));
            CUDA_CHECK(cudaEventElapsedTime(&s2_extract_ms, ws_.s2_extract_start, ws_.s2_extract_end));
            CUDA_CHECK(cudaEventElapsedTime(&s2_d2h_dists_ms, s2_d2h_dists_start, s2_d2h_dists_end));

            std::cout << "[PROFILE] Stage 2 non-batched breakdown (" << num_candidates << " candidates, "
                      << total_tokens << " tokens, k=" << actual_k << "):\n";
            std::cout << "[PROFILE]   1. Gather doc lengths      : " << s2_gather_ms << " ms\n";
            std::cout << "[PROFILE]   2. Prefix sum              : " << s2_prefix_ms << " ms\n";
            std::cout << "[PROFILE]   3. Gather token IDs        : " << s2_tokenids_ms << " ms\n";
            std::cout << "[PROFILE]   4. 1-bit binary IP         : " << s2_binaryip_ms << " ms\n";
            std::cout << "[PROFILE]   5. Doc scoring             : " << s2_docscore_ms << " ms\n";
            std::cout << "[PROFILE]   6. Top-k sort              : " << s2_topk_ms << " ms\n";
            std::cout << "[PROFILE]   7. D2H offsets + indices   : " << s2_d2h_offsets_ms << " ms\n";
            std::cout << "[PROFILE]   8. Extract kernel          : " << s2_extract_ms << " ms\n";
            std::cout << "[PROFILE]   9. D2H final dists + IDs   : " << s2_d2h_dists_ms << " ms\n";
            std::cout << "[PROFILE]   10. Build output on CPU    : " << build_output_time << " ms\n";
            float s2_total = s2_gather_ms + s2_prefix_ms + s2_tokenids_ms + s2_binaryip_ms +
                             s2_docscore_ms + s2_topk_ms + s2_d2h_offsets_ms + s2_extract_ms + s2_d2h_dists_ms + build_output_time;
            std::cout << "[PROFILE]   Sum accounted              : " << s2_total << " ms\n";
            CUDA_CHECK(cudaEventDestroy(s2_topk_start));
            CUDA_CHECK(cudaEventDestroy(s2_topk_end));
            CUDA_CHECK(cudaEventDestroy(s2_d2h_offsets_start));
            CUDA_CHECK(cudaEventDestroy(s2_d2h_offsets_end));
            CUDA_CHECK(cudaEventDestroy(s2_d2h_dists_start));
            CUDA_CHECK(cudaEventDestroy(s2_d2h_dists_end));
        }
#endif

        for (auto doc_id : output_ids) {
            __builtin_prefetch(&doc_ptrs_[doc_id]);
            __builtin_prefetch(&ex_code_[doc_ptrs_[doc_id] * PADDED_DIM * ex_bits / 8]);
        }
    }

// ============================================================
// rank_stage23_persistent: fused Stage 2 + Stage 3 with streaming overlap
// ============================================================
void gpu_mvr_index::rank_stage23_persistent(
        int num_candidates,
        size_t k,
        size_t k_stage2,        // k_rank_all_tokens (e.g., 300)
        query_object* queries,
        std::vector<size_t>& result
    ) {
        rank_stage23_persistent_impl<false>(num_candidates, k, k_stage2, queries, result);
    }

template <bool kProfile>
void gpu_mvr_index::rank_stage23_persistent_impl(
        int num_candidates,
        size_t k,
        size_t k_stage2,        // k_rank_all_tokens (e.g., 300)
        query_object* queries,
        std::vector<size_t>& result
    ) {
        GPU_MVR_NVTX_SCOPE("stage23.rank_stage23_persistent", 0xFF2A9D8F);
        if (num_candidates == 0) return;

        cudaStream_t stream = ws_.stream_compute;
        cudaStream_t stream_d2h = ws_.stream_d2h;
        int actual_k = (int)std::min(k_stage2, (size_t)num_candidates);

        // ========================================
        // Phase A: Data Preparation (GPU)
        // ========================================

        // 1. Gather document lengths
        int threads = 256;
        int blocks = (num_candidates + threads - 1) / threads;

        gather_doc_lengths_kernel<<<blocks, threads, 0, stream>>>(
            ws_.d_topk_doc_ids, d_doc_ptrs_,
            ws_.d_pair_doc_ids,   // reuse as doc_lengths buffer
            num_candidates
        );
        CUDA_CHECK(cudaGetLastError());

        // 2. Prefix sum -> candidate offsets
        CUDA_CHECK(cudaMemsetAsync(ws_.d_pst_candidate_offsets, 0, sizeof(size_t), stream));
        thrust::device_ptr<int> len_ptr(ws_.d_pair_doc_ids);
        thrust::device_ptr<size_t> off_ptr(ws_.d_pst_candidate_offsets + 1);
        auto len_cast = thrust::make_transform_iterator(len_ptr, cast_int_size_t{});
        thrust::inclusive_scan(thrust::cuda::par.on(stream),
                            len_cast, len_cast + num_candidates,
                            off_ptr);

        // 3. Gather token IDs
        gather_token_ids_kernel<<<num_candidates, 256, 0, stream>>>(
            ws_.d_topk_doc_ids,
            d_doc_ptrs_,
            ws_.d_pst_candidate_offsets,
            ws_.d_pst_token_ids,
            num_candidates
        );
        CUDA_CHECK(cudaGetLastError());

        // 4. Copy total_tokens, candidate offsets, and doc IDs to CPU
        size_t total_tokens;
        if constexpr (kProfile) {
            XFER_RECORD_BEGIN(stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(&total_tokens, ws_.d_pst_candidate_offsets + num_candidates,
                                    sizeof(size_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(ws_.v_pst_candidate_offsets.data(), ws_.d_pst_candidate_offsets,
                                    (num_candidates + 1) * sizeof(size_t),
                                    cudaMemcpyDeviceToHost, stream));

        std::vector<int> h_candidate_doc_ids(num_candidates);
        CUDA_CHECK(cudaMemcpyAsync(h_candidate_doc_ids.data(), ws_.d_topk_doc_ids,
                                    num_candidates * sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
        if constexpr (kProfile) {
            XFER_RECORD_END(stream, sizeof(size_t) + (num_candidates + 1) * sizeof(size_t) + num_candidates * sizeof(int), false);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ========================================
        // Phase B: Incrementally launch GPU chunk kernels on stream_compute.
        // The next scoring chunk is not enqueued until after the current chunk's
        // extract kernel has been enqueued on stream_d2h, so extract work is not
        // buried behind later scoring kernels in the GPU scheduler.
        // ========================================

        int score_threads = 128;
        while (score_threads < Q_DOCLEN && score_threads < 256) score_threads *= 2;
        score_threads = std::min(score_threads, 256);

        // Chunk boundaries (candidates are sorted by Stage 1 score descending)
        int actual_chunks = std::min(overlap_chunks, num_candidates);
        int cand_chunk_size = (num_candidates + actual_chunks - 1) / actual_chunks;

        // Create per-chunk compute events
        std::vector<cudaEvent_t> chunk_compute_done(actual_chunks);
        for (int c = 0; c < actual_chunks; c++)
            CUDA_CHECK(cudaEventCreateWithFlags(&chunk_compute_done[c], cudaEventDisableTiming));

        auto launch_chunk = [&](int c) {
            int c_start = c * cand_chunk_size;
            int c_end = std::min(c_start + cand_chunk_size, num_candidates);
            int c_count = c_end - c_start;
            size_t tok_start = ws_.v_pst_candidate_offsets[c_start];
            size_t tok_end = ws_.v_pst_candidate_offsets[c_end];
            size_t tok_count = tok_end - tok_start;

            // binary_ip for this chunk's tokens
            if (tok_count > 0) {
                gpu_mvr::launch_stage2_binary_ip(
                    ws_.d_queries,
#if GPU_MVR_BINARY_IP_IMPL == 2
                    ws_.d_lut,
#else
                    nullptr,
#endif
                    d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
                    ws_.d_pst_token_ids + tok_start,
                    ws_.d_token_dists + tok_start,
                    total_tokens, tok_count, 256, stream
                );
                CUDA_CHECK(cudaGetLastError());
            }

            // doc_score for this chunk's candidates
            gpu_mvr::launch_doc_score(
                ws_.d_token_dists, ws_.d_pst_candidate_offsets + c_start,
                ws_.d_doc_scores + c_start, ws_.d_stage2_doc_query_max,
                total_tokens, c_count, score_threads, stream
            );
            CUDA_CHECK(cudaGetLastError());

            // Record compute completion (gates D2H in Phase C)
            CUDA_CHECK(cudaEventRecord(chunk_compute_done[c], stream));
            if (c == actual_chunks - 1) {
                CUDA_CHECK(cudaEventRecord(ws_.pst_extract_done, stream));
            }
        };

        if (actual_chunks > 0) {
            launch_chunk(0);
        }

        // ========================================
        // Phase C: Streaming CPU processing with GPU-extracted 1-bit dists
        //
        // Per chunk: D2H doc_scores → CPU top-k → GPU extract 1-bit dists
        // → CPU ip_ex_bits (overlapped with GPU extract) → combine → heap
        // ========================================

        double time_wait_d2h = 0;
        double time_running_topk = 0;
        double time_identify_new = 0;
        double time_gpu_extract = 0;
        double time_cpu_ip_ex = 0;
        double time_wait_extract = 0;
        double time_combine = 0;
        std::chrono::high_resolution_clock::time_point t_start_phase_c;
        if constexpr (kProfile) {
            t_start_phase_c = std::chrono::high_resolution_clock::now();
        }

        std::priority_queue<std::pair<float, size_t>> cpu_heap;
        std::unordered_set<int> seen_doc_ids;
        seen_doc_ids.reserve(actual_k * 2);
        int total_refined = 0;

        auto& running_indices = ws_.running_indices;
        auto& h_sel_indices = ws_.h_sel_indices;
        auto& h_out_offsets = ws_.h_out_offsets;
        auto& ip_ex_buf = ws_.ip_ex_buf;
        auto& refined_scores = ws_.refined_scores;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < 1; ++i) {
            (void)i;
        }

        for (int c = 0; c < actual_chunks; c++) {
            // --- D2H doc_scores for this chunk ---
            std::chrono::high_resolution_clock::time_point t0, t1, t2, t3, t4, t5, t6, t7;
            if constexpr (kProfile) {
                t0 = std::chrono::high_resolution_clock::now();
            }
            int c_start = c * cand_chunk_size;
            int c_end = std::min(c_start + cand_chunk_size, num_candidates);
            int c_count = c_end - c_start;
            CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, chunk_compute_done[c], 0));
            if constexpr (kProfile) {
                XFER_RECORD_BEGIN(stream_d2h);
            }
            CUDA_CHECK(cudaMemcpyAsync(
                ws_.h_mapped_doc_scores + c_start,
                ws_.d_doc_scores + c_start,
                c_count * sizeof(float),
                cudaMemcpyDeviceToHost, stream_d2h));
            if constexpr (kProfile) {
                XFER_RECORD_END(stream_d2h, c_count * sizeof(float), false);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream_d2h));
            if constexpr (kProfile) {
                t1 = std::chrono::high_resolution_clock::now();
                time_wait_d2h += std::chrono::duration<double, std::milli>(t1 - t0).count();
            }

            // --- CPU: running top-k from all scores seen so far [0, c_end) ---
            std::iota(running_indices.begin(), running_indices.begin() + c_end, 0);
            int run_k = std::min(actual_k * (c + 1) / actual_chunks, c_end);
            std::nth_element(running_indices.begin(), running_indices.begin() + run_k,
                            running_indices.begin() + c_end,
                            [this](int a, int b) {
                                return ws_.h_mapped_doc_scores[a] > ws_.h_mapped_doc_scores[b];
                            });
            if constexpr (kProfile) {
                t2 = std::chrono::high_resolution_clock::now();
                time_running_topk += std::chrono::duration<double, std::milli>(t2 - t1).count();
            }

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
                if (c + 1 < actual_chunks) {
                    launch_chunk(c + 1);
                }
                continue;
            }
            std::sort(new_docs.begin(), new_docs.end(), std::greater<>());
            int to_refine = new_docs.size();
            new_docs.resize(to_refine);
            for (const auto& p : new_docs) {
                seen_doc_ids.insert(h_candidate_doc_ids[p.second]);
            }
            if constexpr (kProfile) {
                t3 = std::chrono::high_resolution_clock::now();
                time_identify_new += std::chrono::duration<double, std::milli>(t3 - t2).count();
            }

            // --- Prepare selected indices + output offsets for GPU extract ---
            h_out_offsets[0] = 0;
            for (int i = 0; i < to_refine; i++) {
                h_sel_indices[i] = new_docs[i].second;  // candidate index
                int doc_id = h_candidate_doc_ids[h_sel_indices[i]];
                h_out_offsets[i + 1] = h_out_offsets[i] +
                    (size_t)(doc_ptrs_[doc_id + 1] - doc_ptrs_[doc_id]);
            }
            size_t total_sel_tokens = h_out_offsets[to_refine];
            std::vector<gpu_mvr::RefineTask> refine_tasks;
            refine_tasks.reserve(to_refine);
            for (int i = 0; i < to_refine; ++i) {
                const int doc_id = h_candidate_doc_ids[h_sel_indices[i]];
                refine_tasks.push_back(gpu_mvr::RefineTask{
                    i,
                    doc_id,
                    h_out_offsets[i],
                    (size_t)(doc_ptrs_[doc_id + 1] - doc_ptrs_[doc_id])
                });
            }

            // --- Launch GPU extract + D2H (async on stream_d2h) ---
            // H2D of metadata (tiny, synchronous due to pageable memory — negligible)
            if constexpr (kProfile) {
                XFER_RECORD_BEGIN(stream_d2h);
            }
            CUDA_CHECK(cudaMemcpyAsync(ws_.d_selected_indices, h_sel_indices.data(),
                                        to_refine * sizeof(int),
                                        cudaMemcpyHostToDevice, stream_d2h));
            CUDA_CHECK(cudaMemcpyAsync(ws_.d_out_offsets, h_out_offsets.data(),
                                        (to_refine + 1) * sizeof(size_t),
                                        cudaMemcpyHostToDevice, stream_d2h));
            if constexpr (kProfile) {
                XFER_RECORD_END(stream_d2h, to_refine * sizeof(int) + (to_refine + 1) * sizeof(size_t), true);
            }
            // Extract kernel: compacts scattered 1-bit dists into contiguous output
            extract_one_bit_dists_kernel<<<to_refine, 256, 0, stream_d2h>>>(
                ws_.d_token_dists, ws_.d_pst_candidate_offsets,
                ws_.d_selected_indices, ws_.d_out_one_bit_dists,
                ws_.d_out_offsets, total_tokens, (size_t)to_refine
            );
            CUDA_CHECK(cudaGetLastError());
            if constexpr (kProfile) {
                t4 = std::chrono::high_resolution_clock::now();
                time_gpu_extract += std::chrono::duration<double, std::milli>(t4 - t3).count();
            }

            if (c + 1 < actual_chunks) {
                launch_chunk(c + 1);
            }

            // D2H extracted dists (contiguous, async — pinned target)
            if constexpr (kProfile) {
                XFER_RECORD_BEGIN(stream_d2h);
            }
            CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_dists, ws_.d_out_one_bit_dists,
                                        total_sel_tokens * Q_DOCLEN * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream_d2h));
            if constexpr (kProfile) {
                XFER_RECORD_END(stream_d2h, total_sel_tokens * Q_DOCLEN * sizeof(float), false);
            }

            // --- CPU: compute ip_ex_bits (OVERLAPPED with GPU extract + D2H) ---
            // v1 pseudocode:
            //   parallel dynamic for doc in refine_tasks:
            //     for query token, for doc token:
            //       ip_ex_buf[...] = ip_ex_bits(...)
            // v2 pseudocode:
            //   sort docs by token count desc
            //   greedily assign each doc to the least-loaded thread
            //   each thread processes its assigned docs with static work ownership
#if GPU_MVR_CPU_REFINE_IMPL == 1
            gpu_mvr::v1::compute_ip_ex_bits_cpu(*this, queries, refine_tasks, ip_ex_buf.data());
#else
            gpu_mvr::v2::compute_ip_ex_bits_cpu(
                *this, queries, refine_tasks, ws_.h_pinned_queries, ip_ex_buf.data());
#endif
            if constexpr (kProfile) {
                t5 = std::chrono::high_resolution_clock::now();
                time_cpu_ip_ex += std::chrono::duration<double, std::milli>(t5 - t4).count();
            }

            // --- Wait for GPU extract D2H to complete ---
            CUDA_CHECK(cudaStreamSynchronize(stream_d2h));
            if constexpr (kProfile) {
                t6 = std::chrono::high_resolution_clock::now();
                time_wait_extract += std::chrono::duration<double, std::milli>(t6 - t5).count();
            }

            // --- Combine: GPU 1-bit dists + CPU ip_ex_bits → final scores ---
#if GPU_MVR_CPU_REFINE_IMPL == 1
            gpu_mvr::v1::combine_refined_scores_cpu(
                *this, queries, refine_tasks, ws_.h_pinned_dists, ip_ex_buf.data(), refined_scores.data());
#else
            gpu_mvr::v2::combine_refined_scores_cpu(
                *this, queries, refine_tasks, ws_.h_pinned_dists, ip_ex_buf.data(), refined_scores.data());
#endif

            // Add refined results to heap
            for (int i = 0; i < to_refine; i++) {
                cpu_heap.emplace(refined_scores[i].first, (size_t)refined_scores[i].second);
            }
            if constexpr (kProfile) {
                t7 = std::chrono::high_resolution_clock::now();
                time_combine += std::chrono::duration<double, std::milli>(t7 - t6).count();
            }

            total_refined += to_refine;
        }

        if constexpr (kProfile) {
            auto t_end_phase_c = std::chrono::high_resolution_clock::now();
            double time_phase_c = std::chrono::duration<double, std::milli>(t_end_phase_c - t_start_phase_c).count();
            printf("[PROFILE] Phase C: wait_d2h=%.3f ms, topk=%.3f ms, identify=%.3f ms, "
                "gpu_extract=%.3f ms, cpu_ip_ex=%.3f ms, wait_extract=%.3f ms, "
                "combine=%.3f ms (total=%.3f ms, %d docs)\n",
                time_wait_d2h, time_running_topk, time_identify_new,
                time_gpu_extract, time_cpu_ip_ex, time_wait_extract,
                time_combine, time_phase_c, total_refined);
        }

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
    }

    // ======================== STAGE 3: CPU ========================

void gpu_mvr_index::rank_all_tokens_exbits_cpu(
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
        std::vector<gpu_mvr::RefineTask> refine_tasks;
        refine_tasks.reserve(input_ids.size());
        for (size_t idx = 0; idx < input_ids.size(); ++idx) {
            refine_tasks.push_back(gpu_mvr::RefineTask{
                static_cast<int>(idx),
                static_cast<int>(input_ids[idx]),
                candidate_doc_ptrs[idx],
                doc_len(input_ids[idx])
            });
        }

        std::vector<std::pair<float, int>> refined_scores(input_ids.size());
        // v1 pseudocode:
        //   parallel dynamic for doc in refine_tasks:
        //     score doc by scanning all (query_token, doc_token) pairs
        // v2 pseudocode:
        //   sort docs by token count desc
        //   greedily assign each doc to the least-loaded thread
        //   each thread scores only its assigned docs
#if GPU_MVR_CPU_REFINE_IMPL == 1
#pragma omp parallel for schedule(dynamic)
        for (size_t task_idx = 0; task_idx < refine_tasks.size(); ++task_idx) {
            const auto& task = refine_tasks[task_idx];
            float doc_score = 0.0f;
            const size_t doc_start = doc_ptrs_[task.doc_id];
            for (size_t j = 0; j < Q_DOCLEN; ++j) {
                float max_token_score = -std::numeric_limits<float>::infinity();
                for (size_t t = 0; t < task.num_tokens; ++t) {
                    const size_t tid = doc_start + t;
                    const float dist = distance_ex_bits(
                        queries + j,
                        &ex_code_[tid * PADDED_DIM * ex_bits / 8],
                        ex_bits,
                        ip_func_,
                        ws_.h_pinned_dists[(task.out_offset + t) * Q_DOCLEN + j],
                        one_bit_factor_[tid],
                        ex_factor_[tid],
                        PADDED_DIM
                    );
                    max_token_score = std::max(max_token_score, dist);
                }
                doc_score += max_token_score;
            }
            refined_scores[task.local_idx] = {doc_score, task.doc_id};
        }
#else
        {
            const int thread_count = std::min<int>(omp_get_max_threads(), std::max<size_t>(1, refine_tasks.size()));
            const auto assignment = gpu_mvr::v2::build_token_balanced_buckets(refine_tasks, thread_count);
#pragma omp parallel num_threads(thread_count)
            {
                const int tid = omp_get_thread_num();
                for (int task_idx : assignment.buckets[tid]) {
                    const auto& task = refine_tasks[task_idx];
                    float doc_score = 0.0f;
                    const size_t doc_start = doc_ptrs_[task.doc_id];
                    for (size_t j = 0; j < Q_DOCLEN; ++j) {
                        float max_token_score = -std::numeric_limits<float>::infinity();
                        for (size_t t = 0; t < task.num_tokens; ++t) {
                            const size_t tidx = doc_start + t;
                            const float dist = distance_ex_bits(
                                queries + j,
                                &ex_code_[tidx * PADDED_DIM * ex_bits / 8],
                                ex_bits,
                                ip_func_,
                                ws_.h_pinned_dists[(task.out_offset + t) * Q_DOCLEN + j],
                                one_bit_factor_[tidx],
                                ex_factor_[tidx],
                                PADDED_DIM
                            );
                            max_token_score = std::max(max_token_score, dist);
                        }
                        doc_score += max_token_score;
                    }
                    refined_scores[task.local_idx] = {doc_score, task.doc_id};
                }
            }
        }
#endif

        std::priority_queue<std::pair<float, size_t>> max_heap;
        for (const auto& score : refined_scores) {
            max_heap.emplace(score.first, static_cast<size_t>(score.second));
        }
        for (size_t i = 0; i < k && !max_heap.empty(); ++i) {
            output_ids.push_back(max_heap.top().second);
            max_heap.pop();
        }
    }

gpu_mvr_index::~gpu_mvr_index() {
        CUDA_CHECK(cudaDeviceSynchronize());

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
#if GPU_MVR_BINARY_IP_IMPL == 2
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
        CUDA_CHECK(cudaFree(ws_.d_stage2_doc_query_max));
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
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_dists));
#ifndef GPU_MVR_OVERLAP_STAGE23
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_batch_scores));
#endif

#ifdef GPU_MVR_OVERLAP_STAGE23
        // === Persistent Stage 2+3 workspace ===
        CUDA_CHECK(cudaFree(ws_.d_token_dists));
        CUDA_CHECK(cudaFree(ws_.d_pst_candidate_offsets));
        CUDA_CHECK(cudaFree(ws_.d_pst_token_ids));
        CUDA_CHECK(cudaFree(ws_.d_doc_scores));
        CUDA_CHECK(cudaFree(ws_.d_stage2_doc_query_max));
        CUDA_CHECK(cudaFreeHost(ws_.h_mapped_doc_scores));
        CUDA_CHECK(cudaEventDestroy(ws_.pst_compute_done));
        CUDA_CHECK(cudaEventDestroy(ws_.pst_extract_done));
        for (int i = 0; i < Workspace::PST_NUM_D2H_CHUNKS; i++)
            CUDA_CHECK(cudaEventDestroy(ws_.pst_d2h_chunk_done[i]));
#endif

        delete ivf;
        ivf = nullptr;

        // === NEW: Destroy streams and events ===
        CUDA_CHECK(cudaStreamDestroy(ws_.stream_compute));
        CUDA_CHECK(cudaStreamDestroy(ws_.stream_h2d));
        CUDA_CHECK(cudaStreamDestroy(ws_.stream_d2h));
        CUDA_CHECK(cudaEventDestroy(ws_.event_h2d_done));
#if GPU_MVR_STAGE1_SPARSE_CLEANUP == 1
        CUDA_CHECK(cudaEventDestroy(ws_.event_stage1_sum_done));
        CUDA_CHECK(cudaEventDestroy(ws_.event_stage1_cleanup_done));
#endif

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
        rotator_ = nullptr;
}
