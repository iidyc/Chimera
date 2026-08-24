#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>

#include "chimera/chimera_index.cuh"

#define EIGEN_NO_CUDA

#include "chimera/config.cuh"
#include "chimera/ivf_pg.hpp"
#include "chimera/kernels.cuh"
#include "chimera_index_internal.cuh"
#include "rabitqlib/utils/rotator.hpp"

#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace Chimera {

namespace {

struct cast_int_size_t {
    __host__ __device__ size_t operator()(int value) const {
        return static_cast<size_t>(value);
    }
};

constexpr size_t kCandidateRowAlignment = 1u << 16;  // 65536 rows
constexpr size_t kCandidateMinSlackRows = 1u << 15;  // 32768 rows
constexpr size_t kCandidateArenaAlignment = 128;
constexpr size_t kCagraWorkspacePoolFloorBytes = 64ull * 1024ull * 1024ull;

void configure_cagra_workspace(raft::resources& res, size_t estimated_workspace_bytes) {
    constexpr size_t kPoolAlignment = 16ull * 1024ull * 1024ull;
    const size_t aligned_estimate =
        ((estimated_workspace_bytes + kPoolAlignment - 1) / kPoolAlignment) * kPoolAlignment;
    const size_t cagra_pool_bytes = std::max(
        kCagraWorkspacePoolFloorBytes,
        aligned_estimate);
    raft::resource::set_workspace_to_pool_resource(res, cagra_pool_bytes);
}

size_t sum_largest_counts(std::vector<size_t>& counts, size_t top_n) {
    if (counts.empty() || top_n == 0) {
        return 0;
    }

    std::sort(counts.begin(), counts.end(), std::greater<size_t>());
    top_n = std::min(top_n, counts.size());
    return std::accumulate(counts.begin(), counts.begin() + top_n, size_t{0});
}

size_t compute_refined_token_bound(
    const std::vector<int>& doc_ptrs,
    size_t k_refine) {
    if (doc_ptrs.size() < 2 || k_refine == 0) {
        return 0;
    }

    std::vector<size_t> doc_lengths(doc_ptrs.size() - 1, 0);
#pragma omp parallel for schedule(dynamic, 1024)
    for (int doc_idx = 0; doc_idx < static_cast<int>(doc_lengths.size()); ++doc_idx) {
        doc_lengths[doc_idx] =
            static_cast<size_t>(doc_ptrs[doc_idx + 1] - doc_ptrs[doc_idx]);
    }

    std::sort(doc_lengths.begin(), doc_lengths.end(), std::greater<size_t>());

    return std::accumulate(
        doc_lengths.begin(),
        doc_lengths.begin() + std::min(k_refine, doc_lengths.size()),
        size_t{0});
}

size_t align_up(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

char* align_up_ptr(char* ptr, size_t alignment) {
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    const auto aligned = align_up(addr, alignment);
    return reinterpret_cast<char*>(aligned);
}

size_t compute_initial_candidate_capacity(
    size_t num_tokens,
    size_t num_clusters,
    size_t max_nprobe,
    size_t max_candidate_docs) {
    if (max_candidate_docs == 0 || num_clusters == 0 || max_nprobe == 0) {
        return 1;
    }

    const size_t avg_cluster_tokens =
        std::max<size_t>(1, (num_tokens + num_clusters - 1) / num_clusters);

    size_t estimate = avg_cluster_tokens;
    if (estimate > std::numeric_limits<size_t>::max() / max_nprobe) {
        estimate = max_candidate_docs;
    } else {
        estimate *= max_nprobe;
    }

    if (estimate > std::numeric_limits<size_t>::max() / 4) {
        estimate = max_candidate_docs;
    } else {
        estimate *= 4;
    }

    estimate = std::min(max_candidate_docs, estimate);
    estimate = std::min(
        max_candidate_docs, align_up(estimate, kCandidateRowAlignment));
    return std::max<size_t>(estimate, 1);
}

size_t grow_candidate_capacity(size_t current, size_t required, size_t limit) {
    const size_t aligned_required =
        align_up(std::max(required, size_t{1}), kCandidateRowAlignment);
    const size_t required_slack =
        std::max(required / 16, kCandidateMinSlackRows);
    const size_t current_slack =
        std::max(current / 8, kCandidateMinSlackRows);
    size_t target = aligned_required;
    target = std::max(
        target,
        align_up(required + required_slack, kCandidateRowAlignment));
    if (current > 0) {
        target = std::max(
            target,
            align_up(current + current_slack, kCandidateRowAlignment));
    }
    return std::min(limit, target);
}

size_t candidate_arena_bytes(
    size_t row_capacity,
    size_t score_buffer_capacity) {
    size_t offset = 0;
    offset = align_up(offset, kCandidateArenaAlignment);
    offset += row_capacity * sizeof(int);
    offset = align_up(offset, kCandidateArenaAlignment);
    offset += row_capacity * sizeof(int);
    offset = align_up(offset, kCandidateArenaAlignment);
    offset += row_capacity * sizeof(float);
    offset = align_up(offset, kCandidateArenaAlignment);
    offset += row_capacity * Q_DOCLEN * sizeof(float);
    offset = align_up(offset, kCandidateArenaAlignment);
    offset += score_buffer_capacity * sizeof(float);
    offset = align_up(offset, kCandidateArenaAlignment);
    offset += score_buffer_capacity * sizeof(int);
    return offset;
}


int doc_id_from_doc_ptrs_host_binary_range(
    const std::vector<int>& doc_ptrs,
    int lo,
    int hi,
    uint32_t token_id) {
    while (lo < hi) {
        const int mid = lo + ((hi - lo + 1) >> 1);
        const uint32_t mid_start = static_cast<uint32_t>(doc_ptrs[mid]);
        if (mid_start <= token_id) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

int doc_id_from_doc_ptrs_host(
    const std::vector<int>& doc_ptrs,
    uint32_t token_id) {
    return doc_id_from_doc_ptrs_host_binary_range(
        doc_ptrs,
        0,
        static_cast<int>(doc_ptrs.size()) - 2,
        token_id);
}

}  // namespace

void chimera_index::bind_candidate_arena(
    char* base,
    size_t row_capacity,
    size_t score_buffer_capacity) {
    size_t offset = 0;
    offset = align_up(offset, kCandidateArenaAlignment);
    ws_->d_sorted_candidate_doc_ids = reinterpret_cast<int*>(base + offset);
    offset += row_capacity * sizeof(int);
    offset = align_up(offset, kCandidateArenaAlignment);
    ws_->d_materialized_candidate_doc_ids = reinterpret_cast<int*>(base + offset);
    offset += row_capacity * sizeof(int);
    offset = align_up(offset, kCandidateArenaAlignment);
    ws_->d_partial_scores = reinterpret_cast<float*>(base + offset);
    offset += row_capacity * sizeof(float);
    offset = align_up(offset, kCandidateArenaAlignment);
    ws_->d_partial_query_scores = reinterpret_cast<float*>(base + offset);
    offset += row_capacity * Q_DOCLEN * sizeof(float);
    offset = align_up(offset, kCandidateArenaAlignment);
    ws_->d_sorted_partial_scores = reinterpret_cast<float*>(base + offset);
    offset += score_buffer_capacity * sizeof(float);
    offset = align_up(offset, kCandidateArenaAlignment);
    ws_->d_candidate_doc_ids = reinterpret_cast<int*>(base + offset);
}

void chimera_index::compute_candidate_bounds() {
    const size_t probe_cluster_bound =
        std::min(static_cast<size_t>(options_.nprobe), n_clusters);
    max_probed_clusters_ = probe_cluster_bound;
    if (probe_cluster_bound == 0) {
        max_retrieved_tokens_per_query_ = 0;
        max_candidate_docs_per_query_ = 0;
        return;
    }

    std::vector<size_t> cluster_token_counts(n_clusters, 0);
    std::vector<size_t> cluster_unique_doc_counts(n_clusters, 0);

#pragma omp parallel for schedule(dynamic, 256)
    for (int cluster_idx = 0; cluster_idx < static_cast<int>(n_clusters); ++cluster_idx) {
        const size_t cluster_id = static_cast<size_t>(cluster_idx);
        const size_t start = ivf->cluster_pos[cluster_id];
        const size_t end = ivf->cluster_pos[cluster_id + 1];
        const size_t token_count = end - start;
        cluster_token_counts[cluster_id] = token_count;

        if (token_count == 0) {
            cluster_unique_doc_counts[cluster_id] = 0;
            continue;
        }

        const auto first_token_id = static_cast<uint32_t>(ivf->inv_list[start]);
        int prev_doc_id = doc_id_from_doc_ptrs_host(doc_ptrs_, first_token_id);
        size_t unique_doc_count = 1;

        for (size_t pos = start + 1; pos < end; ++pos) {
            const uint32_t token_id = static_cast<uint32_t>(ivf->inv_list[pos]);
            const int current_doc_id = doc_id_from_doc_ptrs_host_binary_range(
                doc_ptrs_, prev_doc_id, static_cast<int>(num_docs) - 1, token_id);
            if (current_doc_id != prev_doc_id) {
                ++unique_doc_count;
                prev_doc_id = current_doc_id;
            }
        }

        cluster_unique_doc_counts[cluster_id] = unique_doc_count;
    }

    max_retrieved_tokens_per_query_ =
        sum_largest_counts(cluster_token_counts, probe_cluster_bound);
    max_candidate_docs_per_query_ =
        sum_largest_counts(cluster_unique_doc_counts, probe_cluster_bound);
}

void chimera_index::ensure_candidate_capacity(size_t required_rows) {
    required_rows = std::max<size_t>(required_rows, 1);
    if (required_rows <= candidate_capacity_ &&
        d_candidate_arena_raw_ != nullptr) {
        return;
    }

    const size_t grown_rows =
        (candidate_capacity_ == 0)
            ? std::min(
                  static_cast<size_t>(num_docs),
                  align_up(required_rows, kCandidateRowAlignment))
            : grow_candidate_capacity(
                  candidate_capacity_, required_rows, static_cast<size_t>(num_docs));
    const size_t score_buffer_capacity =
        std::max(grown_rows, ws_->max_refined_candidates);
    const size_t required_buffer_bytes =
        candidate_arena_bytes(grown_rows, score_buffer_capacity);

    check_cuda(cudaFree(d_candidate_arena_raw_));
    d_candidate_arena_raw_ = nullptr;
    d_candidate_arena_ = nullptr;
    check_cuda(cudaMalloc(
        &d_candidate_arena_raw_,
        required_buffer_bytes + kCandidateArenaAlignment - 1));
    d_candidate_arena_ =
        align_up_ptr(d_candidate_arena_raw_, kCandidateArenaAlignment);
    bind_candidate_arena(
        d_candidate_arena_, grown_rows, score_buffer_capacity);

    candidate_capacity_ = grown_rows;
}

void chimera_index::allocate_workspace() {
    ws_->max_refined_candidates = options_.k_refine;
    ws_->max_full_bit_candidates = options_.k_full_bit;
    ws_->max_refined_tokens =
        compute_refined_token_bound(doc_ptrs_, ws_->max_refined_candidates);
    const size_t max_scored_candidate_docs = std::min(
        static_cast<size_t>(num_docs),
        max_retrieved_tokens_per_query_ * Q_DOCLEN);

    max_candidate_docs_ = std::min(
        static_cast<size_t>(num_docs),
        max_candidate_docs_per_query_ * Q_DOCLEN);
    candidate_bitmap_bucket_count_ = candidate_bitmap_num_buckets(num_docs);
    candidate_bitmap_offset_count_ =
        candidate_bitmap_num_offsets(candidate_bitmap_bucket_count_);
    const size_t initial_candidate_capacity = compute_initial_candidate_capacity(
        n,
        n_clusters,
        max_probed_clusters_,
        max_candidate_docs_);
    const size_t candidate_bound_capacity =
        (max_candidate_docs_per_query_ > 0)
            ? grow_candidate_capacity(
                  0,
                  std::min(max_candidate_docs_per_query_, max_candidate_docs_),
                  max_candidate_docs_)
            : initial_candidate_capacity;
    const size_t candidate_rows =
        std::max(initial_candidate_capacity, candidate_bound_capacity);

    const size_t doc_lengths_bytes =
        ws_->max_refined_candidates * sizeof(int);

    const size_t estimated_size =
        Q_DOCLEN * PADDED_DIM * sizeof(float) +
        Q_DOCLEN * sizeof(float) +
        doc_lengths_bytes +
        candidate_rows * sizeof(int) +
        candidate_rows * sizeof(int) +
        candidate_rows * sizeof(float) +
        candidate_rows * Q_DOCLEN * sizeof(float) +
        candidate_bitmap_bucket_count_ * sizeof(candidate_bitmap_bucket_t) +
        candidate_bitmap_offset_count_ * sizeof(candidate_bitmap_offset_t);
    configure_cagra_workspace(*cagra_res_, estimated_size);

    check_cuda(cudaMalloc(&ws_->d_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
    check_cuda(cudaMalloc(&ws_->d_cb1_sumq, Q_DOCLEN * sizeof(float)));
    check_cuda(cudaMalloc(&ws_->d_refined_doc_lengths, ws_->max_refined_candidates * sizeof(int)));
    check_cuda(cudaMalloc(&ws_->d_refined_doc_offsets,
                          (ws_->max_refined_candidates + 1) * sizeof(size_t)));

    check_cuda(cudaMalloc(&d_candidate_bitmap_,
                          candidate_bitmap_bucket_count_ * sizeof(candidate_bitmap_bucket_t)));
    check_cuda(cudaMalloc(&d_candidate_bitmap_offsets_,
                          candidate_bitmap_offset_count_ * sizeof(candidate_bitmap_offset_t)));

    ws_->d_cub_temp_storage = nullptr;
    ws_->cub_temp_storage_bytes = 0;
    candidate_capacity_ = 0;
    size_t temp_bytes = 0;

    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr,
        temp_bytes,
        ws_->d_partial_scores,
        ws_->d_sorted_partial_scores,
        ws_->d_candidate_doc_ids,
        ws_->d_sorted_candidate_doc_ids,
        static_cast<int>(std::max<size_t>(max_scored_candidate_docs, 1)),
        0,
        32);
    ws_->cub_temp_storage_bytes = std::max(ws_->cub_temp_storage_bytes, temp_bytes);


    {
        thrust::device_ptr<int> len_ptr(ws_->d_refined_doc_lengths);
        auto xform_iter = thrust::make_transform_iterator(len_ptr, cast_int_size_t());
        temp_bytes = 0;
        cub::DeviceScan::InclusiveScan(
            nullptr,
            temp_bytes,
            xform_iter,
            ws_->d_refined_doc_offsets + 1,
            thrust::plus<size_t>(),
            static_cast<int>(ws_->max_refined_candidates));
        ws_->cub_temp_storage_bytes = std::max(ws_->cub_temp_storage_bytes, temp_bytes);
    }

    temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_bytes,
        d_candidate_bitmap_offsets_,
        d_candidate_bitmap_offsets_,
        candidate_bitmap_offset_count_);
    ws_->cub_temp_storage_bytes = std::max(ws_->cub_temp_storage_bytes, temp_bytes);

    check_cuda(cudaMalloc(&ws_->d_cub_temp_storage, ws_->cub_temp_storage_bytes));

    ensure_candidate_capacity(candidate_rows);


    check_cuda(cudaMalloc(&ws_->d_lut, LUT_TOTAL_FLOATS * sizeof(float)));
    const size_t document_scoring_lut_smem =
        DOCUMENT_SCORING_LUT_SMEM_FLOATS * sizeof(float) +
        DOCUMENT_SCORING_LUT_TILE_Q * sizeof(float);
    check_cuda(cudaFuncSetAttribute(
        one_bit_token_score_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        document_scoring_lut_smem));

    check_cuda(cudaMalloc(
        &ws_->d_cagra_dists,
        Q_DOCLEN * max_probed_clusters_ * sizeof(float)));
    check_cuda(cudaMalloc(
        &ws_->d_cagra_labels,
        Q_DOCLEN * max_probed_clusters_ * sizeof(uint32_t)));

    check_cuda(cudaMallocHost(&ws_->h_pinned_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
    check_cuda(cudaMallocHost(&ws_->h_pinned_cb1_sumq, Q_DOCLEN * sizeof(float)));

    check_cuda(cudaMalloc(&ws_->d_refined_token_positions,
                          ws_->max_refined_tokens * sizeof(uint32_t)));
    check_cuda(cudaMalloc(&ws_->d_one_bit_token_scores,
                          ws_->max_refined_tokens * static_cast<size_t>(Q_DOCLEN) * sizeof(float)));
    check_cuda(cudaMalloc(&ws_->d_one_bit_document_scores, ws_->max_refined_candidates * sizeof(float)));

    check_cuda(cudaMallocHost(&ws_->h_mapped_one_bit_document_scores,
                              ws_->max_refined_candidates * sizeof(float)));
    check_cuda(cudaMallocHost(&ws_->h_pinned_refined_doc_offsets,
                              (ws_->max_refined_candidates + 1) * sizeof(size_t)));
    check_cuda(cudaMallocHost(&ws_->h_refined_doc_ids,
                              ws_->max_refined_candidates * sizeof(int)));
    check_cuda(cudaMallocHost(&ws_->h_num_refined_tokens, sizeof(size_t)));

    check_cuda(cudaMemset(
        ws_->d_partial_query_scores,
        0,
        candidate_rows * Q_DOCLEN * sizeof(float)));
    check_cuda(cudaMemset(d_candidate_bitmap_, 0,
                          candidate_bitmap_bucket_count_ * sizeof(candidate_bitmap_bucket_t)));
    check_cuda(cudaMemset(d_candidate_bitmap_offsets_, 0,
                          candidate_bitmap_offset_count_ * sizeof(candidate_bitmap_offset_t)));

    int least_prio, greatest_prio;
    check_cuda(cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio));
    check_cuda(cudaStreamCreateWithPriority(&ws_->stream_compute, cudaStreamDefault, least_prio));
    check_cuda(cudaStreamCreate(&ws_->stream_h2d));
    check_cuda(cudaStreamCreate(&ws_->stream_d2h));
    check_cuda(cudaEventCreate(&ws_->event_h2d_done));


    ws_->one_bit_score_order.resize(ws_->max_refined_candidates);
    ws_->full_bit_candidate_indices.resize(ws_->max_full_bit_candidates);
    ws_->full_bit_scores.resize(ws_->max_full_bit_candidates);
    ws_->seen_doc_map.resize(num_docs, false);
}

void chimera_index::set_doc_mapping(const std::vector<int>& doc_lens) {
    if (doc_lens.empty()) {
        throw std::runtime_error("Chimera index has no documents");
    }
    num_docs = doc_lens.size();
    doc_ptrs_.resize(num_docs + 1, 0);
    for (size_t i = 0; i < num_docs; ++i) {
        if (doc_lens[i] <= 0 ||
            doc_ptrs_[i] > std::numeric_limits<int>::max() - doc_lens[i]) {
            throw std::runtime_error("Invalid document lengths in Chimera index");
        }
        doc_ptrs_[i + 1] = doc_ptrs_[i] + doc_lens[i];
    }
    if (static_cast<size_t>(doc_ptrs_.back()) != n) {
        throw std::runtime_error(
            "Document lengths do not match the number of indexed embeddings");
    }
}

void chimera_index::release_resources() noexcept {
    if (search_ready_) {
        (void)cudaDeviceSynchronize();
    }

    delete ivf;
    ivf = nullptr;
    cagra_res_.reset();

    (void)cudaFree(d_candidate_bitmap_);
    (void)cudaFree(d_candidate_bitmap_offsets_);
    (void)cudaFree(d_candidate_arena_raw_);
    (void)cudaFree(d_doc_ptrs_);
    (void)cudaFree(d_cluster_pos_);
    (void)cudaFree(d_clustered_one_bit_codes_);
    (void)cudaFree(d_clustered_one_bit_factors_);
    (void)cudaFree(d_clustered_doc_ids_);
    (void)cudaFree(d_token_to_cluster_pos_);

    if (ws_ != nullptr) {
        (void)cudaFree(ws_->d_queries);
        (void)cudaFree(ws_->d_cb1_sumq);
        (void)cudaFree(ws_->d_lut);
        (void)cudaFree(ws_->d_refined_doc_lengths);
        (void)cudaFree(ws_->d_cagra_dists);
        (void)cudaFree(ws_->d_cagra_labels);
        (void)cudaFree(ws_->d_cub_temp_storage);
        (void)cudaFreeHost(ws_->h_pinned_queries);
        (void)cudaFreeHost(ws_->h_pinned_cb1_sumq);
        (void)cudaFree(ws_->d_one_bit_token_scores);
        (void)cudaFree(ws_->d_refined_doc_offsets);
        (void)cudaFree(ws_->d_refined_token_positions);
        (void)cudaFree(ws_->d_one_bit_document_scores);
        (void)cudaFreeHost(ws_->h_mapped_one_bit_document_scores);
        (void)cudaFreeHost(ws_->h_pinned_refined_doc_offsets);
        (void)cudaFreeHost(ws_->h_refined_doc_ids);
        (void)cudaFreeHost(ws_->h_num_refined_tokens);
        (void)cudaStreamDestroy(ws_->stream_compute);
        (void)cudaStreamDestroy(ws_->stream_h2d);
        (void)cudaStreamDestroy(ws_->stream_d2h);
        (void)cudaEventDestroy(ws_->event_h2d_done);
    }
    ws_.reset();

    delete rotator_;
    rotator_ = nullptr;

    d_candidate_bitmap_ = nullptr;
    d_candidate_bitmap_offsets_ = nullptr;
    d_candidate_arena_raw_ = nullptr;
    d_candidate_arena_ = nullptr;
    d_doc_ptrs_ = nullptr;
    d_cluster_pos_ = nullptr;
    d_clustered_one_bit_codes_ = nullptr;
    d_clustered_one_bit_factors_ = nullptr;
    d_clustered_doc_ids_ = nullptr;
    d_token_to_cluster_pos_ = nullptr;

    full_bit_codes_.clear();
    full_bit_factors_.clear();
    one_bit_codes_.clear();
    one_bit_factors_.clear();
    clustered_data_.clear();
    doc_lens_.clear();
    doc_ptrs_.clear();
    n = 0;
    d = 0;
    n_clusters = 0;
    ex_bits = 0;
    num_docs = 0;
    max_cluster_size = 0;
    has_index_ = false;
    search_ready_ = false;
}

}  // namespace Chimera
