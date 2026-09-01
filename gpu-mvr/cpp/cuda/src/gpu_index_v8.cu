#include "gpu_index_v8.cuh"

#ifdef GPU_MVR_HAVE_CUVS
#include <raft/core/resource/device_memory_resource.hpp>
#endif

#define GPU_MVR_V3_SKIP_WORKSPACE_ALLOC 1
#define gpu_mvr_index gpu_mvr_index_v8_base
#define cast_int_size_t cast_int_size_t_v8_base
#include "gpu_index_v3.cu"
#undef cast_int_size_t
#undef gpu_mvr_index
#undef GPU_MVR_V3_SKIP_WORKSPACE_ALLOC

#include <cub/cub.cuh>
#include <iostream>

namespace {

constexpr size_t kV8CompactDocRowAlignment = 1u << 16;  // 65536 rows
constexpr size_t kV8CompactDocMinSlackRows = 1u << 15;  // 32768 rows
constexpr size_t kV8CompactDocBufferAlignment = 128;
constexpr size_t kV8CagraWorkspacePoolFloorBytes = 64ull * 1024ull * 1024ull;

#ifdef GPU_MVR_HAVE_CUVS
void v8_configure_cagra_workspace(raft::resources& res, size_t estimated_workspace_bytes) {
    constexpr size_t kPoolAlignment = 16ull * 1024ull * 1024ull;
    const size_t aligned_estimate =
        ((estimated_workspace_bytes + kPoolAlignment - 1) / kPoolAlignment) * kPoolAlignment;
    const size_t cagra_pool_bytes = std::max(
        kV8CagraWorkspacePoolFloorBytes,
        aligned_estimate);
    raft::resource::set_workspace_to_pool_resource(res, cagra_pool_bytes);
}
#endif

size_t v8_sum_top_sorted_counts(std::vector<size_t>& counts, size_t top_n) {
    if (counts.empty() || top_n == 0) {
        return 0;
    }

    std::sort(counts.begin(), counts.end(), std::greater<size_t>());
    top_n = std::min(top_n, counts.size());
    return std::accumulate(counts.begin(), counts.begin() + top_n, size_t{0});
}

std::pair<size_t, size_t> v8_compute_stage2_token_bounds(
    const std::vector<int>& doc_ptrs,
    size_t top_candidates,
    size_t topk) {
    if (doc_ptrs.size() < 2 || (top_candidates == 0 && topk == 0)) {
        return {0, 0};
    }

    std::vector<size_t> doc_lengths(doc_ptrs.size() - 1, 0);
#pragma omp parallel for schedule(dynamic, 1024)
    for (int doc_idx = 0; doc_idx < static_cast<int>(doc_lengths.size()); ++doc_idx) {
        doc_lengths[doc_idx] =
            static_cast<size_t>(doc_ptrs[doc_idx + 1] - doc_ptrs[doc_idx]);
    }

    std::sort(doc_lengths.begin(), doc_lengths.end(), std::greater<size_t>());

    const size_t candidate_limit = std::min(top_candidates, doc_lengths.size());
    const size_t topk_limit = std::min(topk, doc_lengths.size());
    const size_t max_limit = std::max(candidate_limit, topk_limit);

    size_t prefix_sum = 0;
    size_t candidate_tokens = 0;
    size_t topk_tokens = 0;
    for (size_t idx = 0; idx < max_limit; ++idx) {
        prefix_sum += doc_lengths[idx];
        if (idx + 1 == candidate_limit) {
            candidate_tokens = prefix_sum;
        }
        if (idx + 1 == topk_limit) {
            topk_tokens = prefix_sum;
        }
    }

    if (candidate_limit == 0) {
        candidate_tokens = 0;
    }
    if (topk_limit == 0) {
        topk_tokens = 0;
    }

    return {candidate_tokens, topk_tokens};
}

size_t v8_align_up(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

char* v8_align_up_ptr(char* ptr, size_t alignment) {
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    const auto aligned = v8_align_up(addr, alignment);
    return reinterpret_cast<char*>(aligned);
}

size_t v8_initial_compact_doc_capacity(
    size_t num_tokens,
    size_t num_clusters,
    size_t max_nprobe,
    size_t max_compact_docs) {
    if (max_compact_docs == 0 || num_clusters == 0 || max_nprobe == 0) {
        return 1;
    }

    const size_t avg_cluster_tokens =
        std::max<size_t>(1, (num_tokens + num_clusters - 1) / num_clusters);

    size_t estimate = avg_cluster_tokens;
    if (estimate > std::numeric_limits<size_t>::max() / max_nprobe) {
        estimate = max_compact_docs;
    } else {
        estimate *= max_nprobe;
    }

    if (estimate > std::numeric_limits<size_t>::max() / 4) {
        estimate = max_compact_docs;
    } else {
        estimate *= 4;
    }

    estimate = std::min(max_compact_docs, estimate);
    estimate = std::min(max_compact_docs, v8_align_up(estimate, kV8CompactDocRowAlignment));
    return std::max<size_t>(estimate, 1);
}

size_t v8_grow_compact_doc_capacity(size_t current, size_t required, size_t limit) {
    const size_t aligned_required =
        v8_align_up(std::max(required, size_t{1}), kV8CompactDocRowAlignment);
    const size_t required_slack =
        std::max(required / 16, kV8CompactDocMinSlackRows);
    const size_t current_slack =
        std::max(current / 8, kV8CompactDocMinSlackRows);
    size_t target = aligned_required;
    target = std::max(
        target,
        v8_align_up(required + required_slack, kV8CompactDocRowAlignment));
    if (current > 0) {
        target = std::max(
            target,
            v8_align_up(current + current_slack, kV8CompactDocRowAlignment));
    }
    return std::min(limit, target);
}

size_t v8_compact_doc_arena_bytes(size_t row_capacity, size_t topk_capacity) {
    size_t offset = 0;
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    offset += row_capacity * sizeof(int);
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    offset += row_capacity * sizeof(int);
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    offset += row_capacity * sizeof(float);
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    offset += row_capacity * Q_DOCLEN * sizeof(float);
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    offset += topk_capacity * sizeof(float);
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    offset += topk_capacity * sizeof(int);
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    offset += topk_capacity * sizeof(int);
    return offset;
}

void v8_bind_compact_doc_arena(
    gpu_mvr_index_v8_base::Workspace& ws,
    char* base,
    size_t row_capacity,
    size_t topk_capacity) {
    size_t offset = 0;
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    ws.d_sorted_doc_ids = reinterpret_cast<int*>(base + offset);
    offset += row_capacity * sizeof(int);
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    ws.d_unique_doc_ids = reinterpret_cast<int*>(base + offset);
    offset += row_capacity * sizeof(int);
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    ws.d_stage1_doc_scores = reinterpret_cast<float*>(base + offset);
    offset += row_capacity * sizeof(float);
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    ws.d_doc_query_max = reinterpret_cast<float*>(base + offset);
    offset += row_capacity * Q_DOCLEN * sizeof(float);
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    ws.d_topk_scores = reinterpret_cast<float*>(base + offset);
    offset += topk_capacity * sizeof(float);
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    ws.d_topk_doc_ids = reinterpret_cast<int*>(base + offset);
    offset += topk_capacity * sizeof(int);
    offset = v8_align_up(offset, kV8CompactDocBufferAlignment);
    ws.d_topk_indices = reinterpret_cast<int*>(base + offset);
}

int v8_doc_id_from_doc_ptrs_host_binary_range(
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

int v8_doc_id_from_doc_ptrs_host(
    const std::vector<int>& doc_ptrs,
    uint32_t token_id) {
    return v8_doc_id_from_doc_ptrs_host_binary_range(
        doc_ptrs,
        0,
        static_cast<int>(doc_ptrs.size()) - 2,
        token_id);
}

}  // namespace

gpu_mvr_index::gpu_mvr_index(
    const std::string& filename,
    const std::vector<int>& doc_lens,
    const gpu_search_runtime_options& runtime_options)
    : gpu_mvr_index_v8_base(filename, doc_lens, runtime_options) {
    std::cout << "[v8] Stage1 align mode: compact_align with shifted cluster loads\n";
    compute_workspace_probe_bounds();
    allocate_workspace();
}

gpu_mvr_index::gpu_mvr_index(
    gpu_mvr_index& owner,
    const gpu_search_runtime_options& runtime_options)
    : gpu_mvr_index_v8_base() {
    is_query_slot_ = true;

    n = owner.n;
    d = owner.d;
    n_clusters = owner.n_clusters;
    ex_bits = owner.ex_bits;
    num_docs = owner.num_docs;
    max_doc_len = owner.max_doc_len;
    max_cluster_size = owner.max_cluster_size;
    workspace_probe_cluster_bound_ = owner.workspace_probe_cluster_bound_;
    workspace_probe_token_bound_ = owner.workspace_probe_token_bound_;
    workspace_probe_unique_doc_bound_ = owner.workspace_probe_unique_doc_bound_;

    rotator_ = owner.rotator_;
    ivf = owner.ivf;
    doc_ptrs_ = owner.doc_ptrs_;
    full_code_source_ = owner.full_code_source_;
    ex_factor_source_ = owner.ex_factor_source_;
    ip_func_ = owner.ip_func_;
    unpack_func_ = owner.unpack_func_;

    d_doc_ptrs_ = owner.d_doc_ptrs_;
    d_inv_list_ = owner.d_inv_list_;
    d_cluster_pos_ = owner.d_cluster_pos_;
    d_clustered_code_ = owner.d_clustered_code_;
    d_clustered_factor_ = owner.d_clustered_factor_;
    d_clustered_doc_ids_ = owner.d_clustered_doc_ids_;
    d_token_to_cluster_pos_ = owner.d_token_to_cluster_pos_;
#ifdef GPU_MVR_STAGE2_DOC_LAYOUT
    d_one_bit_code_ = owner.d_one_bit_code_;
    d_one_bit_factor_ = owner.d_one_bit_factor_;
#endif
    use_clustered_ = owner.use_clustered_;

    nprobe = runtime_options.nprobe;
    k_rank_cluster = runtime_options.k_rank_cluster;
    k_rank_all_tokens = runtime_options.k_rank_all_tokens;
    itopk_size = runtime_options.itopk_size;
    overlap_chunks = runtime_options.overlap_chunks;

    allocate_workspace();
}

gpu_mvr_index::~gpu_mvr_index() {
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_doc_bitmap_));
    CUDA_CHECK(cudaFree(d_doc_bitmap_offsets_));
    CUDA_CHECK(cudaFree(d_compact_doc_buffer_raw_));
    d_doc_bitmap_ = nullptr;
    d_doc_bitmap_offsets_ = nullptr;
    d_compact_doc_buffer_raw_ = nullptr;
    d_compact_doc_buffer_ = nullptr;

    ws_.d_sorted_doc_ids = nullptr;
    ws_.d_unique_doc_ids = nullptr;
    ws_.d_stage1_doc_scores = nullptr;
    ws_.d_doc_query_max = nullptr;
    ws_.d_topk_scores = nullptr;
    ws_.d_topk_doc_ids = nullptr;
    ws_.d_topk_indices = nullptr;
    ws_.d_doc_touched = nullptr;

    if (is_query_slot_) {
        ivf = nullptr;
        rotator_ = nullptr;
        d_doc_ptrs_ = nullptr;
        d_inv_list_ = nullptr;
        d_cluster_pos_ = nullptr;
        d_clustered_code_ = nullptr;
        d_clustered_factor_ = nullptr;
        d_clustered_doc_ids_ = nullptr;
        d_token_to_cluster_pos_ = nullptr;
#ifdef GPU_MVR_STAGE2_DOC_LAYOUT
        d_one_bit_code_ = nullptr;
        d_one_bit_factor_ = nullptr;
#endif
    }
}

void gpu_mvr_index::compute_workspace_probe_bounds() {
    const size_t probe_cluster_bound =
        std::min(static_cast<size_t>(std::max(nprobe, 0)), n_clusters);
    workspace_probe_cluster_bound_ = probe_cluster_bound;
    if (probe_cluster_bound == 0) {
        workspace_probe_token_bound_ = 0;
        workspace_probe_unique_doc_bound_ = 0;
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
        int prev_doc_id = v8_doc_id_from_doc_ptrs_host(doc_ptrs_, first_token_id);
        size_t unique_doc_count = 1;

        for (size_t pos = start + 1; pos < end; ++pos) {
            const uint32_t token_id = static_cast<uint32_t>(ivf->inv_list[pos]);
            const int current_doc_id = v8_doc_id_from_doc_ptrs_host_binary_range(
                doc_ptrs_, prev_doc_id, static_cast<int>(num_docs) - 1, token_id);
            if (current_doc_id != prev_doc_id) {
                ++unique_doc_count;
                prev_doc_id = current_doc_id;
            }
        }

        cluster_unique_doc_counts[cluster_id] = unique_doc_count;
    }

    workspace_probe_token_bound_ =
        v8_sum_top_sorted_counts(cluster_token_counts, probe_cluster_bound);
    workspace_probe_unique_doc_bound_ =
        v8_sum_top_sorted_counts(cluster_unique_doc_counts, probe_cluster_bound);
}

void gpu_mvr_index::ensure_compact_doc_capacity(size_t required_rows) {
    required_rows = std::max<size_t>(required_rows, 1);
    if (required_rows <= compact_doc_capacity_ &&
        d_compact_doc_buffer_raw_ != nullptr) {
        return;
    }

    const size_t grown_rows =
        (compact_doc_capacity_ == 0)
            ? std::min(
                  static_cast<size_t>(num_docs),
                  v8_align_up(required_rows, kV8CompactDocRowAlignment))
            : v8_grow_compact_doc_capacity(
                  compact_doc_capacity_, required_rows, static_cast<size_t>(num_docs));
    const size_t required_topk_capacity =
        std::max(grown_rows, ws_.max_stage2_candidates);
    const size_t required_buffer_bytes =
        v8_compact_doc_arena_bytes(grown_rows, required_topk_capacity);

    std::cout << "[workspace][v8] Growing compact doc capacity from "
              << compact_doc_capacity_ << " to " << grown_rows
              << " rows (required=" << required_rows << ")" << std::endl;

    CUDA_CHECK(cudaFree(d_compact_doc_buffer_raw_));
    d_compact_doc_buffer_raw_ = nullptr;
    d_compact_doc_buffer_ = nullptr;
    compact_doc_buffer_bytes_ = required_buffer_bytes;
    CUDA_CHECK(cudaMalloc(
        &d_compact_doc_buffer_raw_,
        compact_doc_buffer_bytes_ + kV8CompactDocBufferAlignment - 1));
    d_compact_doc_buffer_ =
        v8_align_up_ptr(d_compact_doc_buffer_raw_, kV8CompactDocBufferAlignment);
    v8_bind_compact_doc_arena(ws_, d_compact_doc_buffer_, grown_rows, required_topk_capacity);

    compact_doc_capacity_ = grown_rows;
    compact_topk_capacity_ = required_topk_capacity;
}

void accumulate_gpu_search_profile_v8(gpu_search_profile_v8& total, const gpu_search_profile_v8& sample) {
    total.search_wall_ms += sample.search_wall_ms;
    total.total_search_time_ms += sample.total_search_time_ms;
    total.stage1_time_ms += sample.stage1_time_ms;
    total.stage2_time_ms += sample.stage2_time_ms;
    total.s1_cagra_ms += sample.s1_cagra_ms;
    total.s1_expansion_ms += sample.s1_expansion_ms;
    total.s1_binary_ip_ms += sample.s1_binary_ip_ms;
    total.s1_atomic_agg_ms += sample.s1_atomic_agg_ms;
    total.s1_sum_scores_ms += sample.s1_sum_scores_ms;
    total.s1_topk_sort_ms += sample.s1_topk_sort_ms;
    total.s1_d2d_ms += sample.s1_d2d_ms;
    total.s1_memset_ms += sample.s1_memset_ms;
    total.s1_sum_accounted_ms += sample.s1_sum_accounted_ms;
    total.phase_a_wall_ms += sample.phase_a_wall_ms;
    total.phase_a_gpu_total_ms += sample.phase_a_gpu_total_ms;
    total.phase_a_gather_ms += sample.phase_a_gather_ms;
    total.phase_a_prefix_ms += sample.phase_a_prefix_ms;
    total.phase_a_token_ids_ms += sample.phase_a_token_ids_ms;
    total.phase_a_d2h_ms += sample.phase_a_d2h_ms;
    total.phase_a_cpu_launch_gather_ms += sample.phase_a_cpu_launch_gather_ms;
    total.phase_a_cpu_launch_prefix_ms += sample.phase_a_cpu_launch_prefix_ms;
    total.phase_a_cpu_launch_token_ids_ms += sample.phase_a_cpu_launch_token_ids_ms;
    total.phase_a_cpu_launch_d2h_ms += sample.phase_a_cpu_launch_d2h_ms;
    total.phase_a_cpu_sync_ms += sample.phase_a_cpu_sync_ms;
    total.phase_a_cpu_event_elapsed_ms += sample.phase_a_cpu_event_elapsed_ms;
    total.phase_a_gpu_sum_accounted_ms += sample.phase_a_gpu_sum_accounted_ms;
    total.phase_b_wall_ms += sample.phase_b_wall_ms;
    total.phase_b_binary_ip_total_ms += sample.phase_b_binary_ip_total_ms;
    total.phase_b_doc_score_total_ms += sample.phase_b_doc_score_total_ms;
    total.phase_b_total_kernel_ms += sample.phase_b_total_kernel_ms;
    total.phase_c_wait_d2h_ms += sample.phase_c_wait_d2h_ms;
    total.phase_c_topk_ms += sample.phase_c_topk_ms;
    total.phase_c_identify_ms += sample.phase_c_identify_ms;
    total.phase_c_prepare_ms += sample.phase_c_prepare_ms;
    total.phase_c_cpu_refine_ms += sample.phase_c_cpu_refine_ms;
    total.phase_c_combine_ms += sample.phase_c_combine_ms;
    total.phase_c_total_ms += sample.phase_c_total_ms;
    total.phase_c_refined_docs += sample.phase_c_refined_docs;
    total.phase_abc_total_wall_ms += sample.phase_abc_total_wall_ms;
    total.transfer_h2d_ms += sample.transfer_h2d_ms;
    total.transfer_d2h_ms += sample.transfer_d2h_ms;
    total.transfer_total_ms += sample.transfer_total_ms;
    total.transfer_h2d_bytes += sample.transfer_h2d_bytes;
    total.transfer_d2h_bytes += sample.transfer_d2h_bytes;
    total.transfer_count += sample.transfer_count;
}

void average_gpu_search_profile_v8(gpu_search_profile_v8& profile, double count) {
    if (count <= 0.0) return;
    profile.search_wall_ms /= count;
    profile.total_search_time_ms /= count;
    profile.stage1_time_ms /= count;
    profile.stage2_time_ms /= count;
    profile.s1_cagra_ms /= count;
    profile.s1_expansion_ms /= count;
    profile.s1_binary_ip_ms /= count;
    profile.s1_atomic_agg_ms /= count;
    profile.s1_sum_scores_ms /= count;
    profile.s1_topk_sort_ms /= count;
    profile.s1_d2d_ms /= count;
    profile.s1_memset_ms /= count;
    profile.s1_sum_accounted_ms /= count;
    profile.phase_a_wall_ms /= count;
    profile.phase_a_gpu_total_ms /= count;
    profile.phase_a_gather_ms /= count;
    profile.phase_a_prefix_ms /= count;
    profile.phase_a_token_ids_ms /= count;
    profile.phase_a_d2h_ms /= count;
    profile.phase_a_cpu_launch_gather_ms /= count;
    profile.phase_a_cpu_launch_prefix_ms /= count;
    profile.phase_a_cpu_launch_token_ids_ms /= count;
    profile.phase_a_cpu_launch_d2h_ms /= count;
    profile.phase_a_cpu_sync_ms /= count;
    profile.phase_a_cpu_event_elapsed_ms /= count;
    profile.phase_a_gpu_sum_accounted_ms /= count;
    profile.phase_b_wall_ms /= count;
    profile.phase_b_binary_ip_total_ms /= count;
    profile.phase_b_doc_score_total_ms /= count;
    profile.phase_b_total_kernel_ms /= count;
    profile.phase_c_wait_d2h_ms /= count;
    profile.phase_c_topk_ms /= count;
    profile.phase_c_identify_ms /= count;
    profile.phase_c_prepare_ms /= count;
    profile.phase_c_cpu_refine_ms /= count;
    profile.phase_c_combine_ms /= count;
    profile.phase_c_total_ms /= count;
    profile.phase_c_refined_docs /= count;
    profile.phase_abc_total_wall_ms /= count;
    profile.transfer_h2d_ms /= count;
    profile.transfer_d2h_ms /= count;
    profile.transfer_total_ms /= count;
    profile.transfer_h2d_bytes /= count;
    profile.transfer_d2h_bytes /= count;
    profile.transfer_count /= count;
}

void print_gpu_search_profile_v8(const gpu_search_profile_v8& p, const char* prefix) {
    std::cout << prefix << " Mode: V8 persistent Stage 2+3\n";
    std::cout << prefix << " Search wall-clock time      : " << p.search_wall_ms << " ms\n";
    std::cout << prefix << " Total GPU search time       : " << p.total_search_time_ms << " ms\n";
    std::cout << prefix << " Stage 1 time                : " << p.stage1_time_ms << " ms\n";
    std::cout << prefix << "   1. CAGRA search            : " << p.s1_cagra_ms << " ms\n";
    std::cout << prefix << "   2. GPU IVF expansion       : " << p.s1_expansion_ms << " ms\n";
    std::cout << prefix << "   3. Binary IP kernel        : " << p.s1_binary_ip_ms << " ms\n";
    std::cout << prefix << "   4. Aggregation + tracking  : " << p.s1_atomic_agg_ms << " ms\n";
    std::cout << prefix << "   5. Sum doc scores (sparse) : " << p.s1_sum_scores_ms << " ms\n";
    std::cout << prefix << "   6. Top-k sort (sparse)     : " << p.s1_topk_sort_ms << " ms\n";
    std::cout << prefix << "   7. D2D copy top-k doc IDs  : " << p.s1_d2d_ms << " ms\n";
    std::cout << prefix << "   8. Memset (overlapped 1-3) : " << p.s1_memset_ms << " ms (not in critical path)\n";
    std::cout << prefix << "   Sum accounted              : " << p.s1_sum_accounted_ms << " ms\n";
    std::cout << prefix << " Stage 2+3 event time        : " << p.stage2_time_ms << " ms\n";
    std::cout << prefix << " Phase A (Data Preparation) wall: " << p.phase_a_wall_ms
              << " ms, GPU total: " << p.phase_a_gpu_total_ms << " ms\n";
    std::cout << prefix << "   1. Gather doc lengths      : " << p.phase_a_gather_ms
              << " ms (CPU launch: " << p.phase_a_cpu_launch_gather_ms << " ms)\n";
    std::cout << prefix << "   2. Prefix sum offsets      : " << p.phase_a_prefix_ms
              << " ms (CPU launch: " << p.phase_a_cpu_launch_prefix_ms << " ms)\n";
    std::cout << prefix << "   3. Gather clustered pos    : " << p.phase_a_token_ids_ms
              << " ms (CPU launch: " << p.phase_a_cpu_launch_token_ids_ms << " ms)\n";
    std::cout << prefix << "   4. D2H metadata + sync     : " << p.phase_a_d2h_ms
              << " ms (CPU launch: " << p.phase_a_cpu_launch_d2h_ms << " ms)\n";
    std::cout << prefix << "   5. cudaStreamSynchronize   : " << p.phase_a_cpu_sync_ms << " ms\n";
    std::cout << prefix << "   6. cudaEventElapsedTime x5 : " << p.phase_a_cpu_event_elapsed_ms << " ms\n";
    std::cout << prefix << "   GPU sum accounted           : " << p.phase_a_gpu_sum_accounted_ms << " ms\n";
    std::cout << prefix << " Phase B wall time           : " << p.phase_b_wall_ms << " ms\n";
    std::cout << prefix << " Phase B binary_ip total     : " << p.phase_b_binary_ip_total_ms << " ms\n";
    std::cout << prefix << " Phase B doc_score total     : " << p.phase_b_doc_score_total_ms << " ms\n";
    std::cout << prefix << " Phase B total kernel time   : " << p.phase_b_total_kernel_ms << " ms\n";
    std::cout << prefix << " Phase C: wait_d2h=" << p.phase_c_wait_d2h_ms
              << " ms, topk=" << p.phase_c_topk_ms
              << " ms, identify=" << p.phase_c_identify_ms
              << " ms, prepare=" << p.phase_c_prepare_ms
              << " ms, cpu_refine=" << p.phase_c_cpu_refine_ms
              << " ms, combine=" << p.phase_c_combine_ms
              << " ms (total=" << p.phase_c_total_ms
              << " ms, docs=" << p.phase_c_refined_docs << ")\n";
    std::cout << prefix << " Total wall time for Phase A + B + C: "
              << p.phase_abc_total_wall_ms << " ms\n";
    std::cout << prefix << " Data transfer summary       : H2D=" << p.transfer_h2d_ms
              << " ms (" << p.transfer_h2d_bytes << " bytes), D2H=" << p.transfer_d2h_ms
              << " ms (" << p.transfer_d2h_bytes << " bytes), total="
              << p.transfer_total_ms << " ms, count=" << p.transfer_count << "\n";
}

void gpu_mvr_index::allocate_workspace() {
    ws_.max_q_doclen = Q_DOCLEN;
    ws_.max_stage1_pairs = workspace_probe_token_bound_ * Q_DOCLEN;
    ws_.max_stage2_candidates = k_rank_cluster;
    ws_.max_stage2_k = k_rank_all_tokens;
    std::tie(ws_.max_stage2_tokens, ws_.max_stage2_k_tokens) =
        v8_compute_stage2_token_bounds(doc_ptrs_, ws_.max_stage2_candidates, ws_.max_stage2_k);
    ws_.estimated_num_docs = static_cast<size_t>(num_docs);
    ws_.max_stage1_touched_docs =
        std::min(ws_.estimated_num_docs, ws_.max_stage1_pairs);

    max_compact_docs_ = std::min(
        static_cast<size_t>(num_docs),
        workspace_probe_unique_doc_bound_ * Q_DOCLEN);
    doc_bitmap_bucket_count_ = doc_bitmap_num_buckets(num_docs);
    doc_bitmap_offset_count_ = doc_bitmap_num_offsets(doc_bitmap_bucket_count_);
    const size_t initial_compact_doc_capacity = v8_initial_compact_doc_capacity(
        n,
        n_clusters,
        workspace_probe_cluster_bound_,
        max_compact_docs_);
    const size_t compact_doc_bound_capacity =
        (workspace_probe_unique_doc_bound_ > 0)
            ? v8_grow_compact_doc_capacity(
                  0,
                  std::min(workspace_probe_unique_doc_bound_, max_compact_docs_),
                  max_compact_docs_)
            : initial_compact_doc_capacity;
    const size_t doc_buf_rows =
        std::max(initial_compact_doc_capacity, compact_doc_bound_capacity);

    std::cout << "max cluster size: " << max_cluster_size << std::endl;
    std::cout << "workspace_probe_cluster_bound: "
              << workspace_probe_cluster_bound_ << std::endl;
    std::cout << "workspace_probe_token_bound_per_query: "
              << workspace_probe_token_bound_ << std::endl;
    std::cout << "workspace_probe_unique_doc_bound_per_query: "
              << workspace_probe_unique_doc_bound_ << std::endl;
    std::cout << "max_compact_docs: " << max_compact_docs_
              << "  (num_docs=" << num_docs << ")" << std::endl;
    std::cout << "initial_compact_doc_capacity: "
              << initial_compact_doc_capacity << std::endl;
    std::cout << "preallocated_compact_doc_capacity: "
              << doc_buf_rows << std::endl;
    std::cout << "workspace_stage2_candidate_token_bound: "
              << ws_.max_stage2_tokens << std::endl;
    std::cout << "workspace_stage2_topk_token_bound: "
              << ws_.max_stage2_k_tokens << std::endl;

    const size_t stage1_emb_ids_bytes = 0;
    const size_t stage1_pair_offsets_bytes = 0;
    const size_t stage1_emb_dists_bytes = 0;
    const size_t phase_a_doc_lengths_bytes =
        ws_.max_stage2_candidates * sizeof(int);

    const size_t estimated_size =
        Q_DOCLEN * PADDED_DIM * sizeof(float) +
        Q_DOCLEN * sizeof(float) +
        stage1_emb_ids_bytes +
        stage1_pair_offsets_bytes +
        stage1_emb_dists_bytes +
        phase_a_doc_lengths_bytes +
        doc_buf_rows * sizeof(int) +
        doc_buf_rows * sizeof(int) +
        doc_buf_rows * sizeof(float) +
        doc_buf_rows * Q_DOCLEN * sizeof(float) +
        doc_bitmap_bucket_count_ * sizeof(doc_bitmap_bucket_t) +
        doc_bitmap_offset_count_ * sizeof(doc_bitmap_offset_t);
    std::cout << "Initial GPU memory required for v8 workspace: "
              << (estimated_size / (1024.0 * 1024.0)) << " MB" << std::endl;
#ifdef GPU_MVR_HAVE_CUVS
    v8_configure_cagra_workspace(cagra_res_, estimated_size);
#endif

    CUDA_CHECK(cudaMalloc(&ws_.d_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_cb1_sumq, Q_DOCLEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_doc_ids, ws_.max_stage2_candidates * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_doc_bitmap_,
                          doc_bitmap_bucket_count_ * sizeof(doc_bitmap_bucket_t)));
    CUDA_CHECK(cudaMalloc(&d_doc_bitmap_offsets_,
                          doc_bitmap_offset_count_ * sizeof(doc_bitmap_offset_t)));

    ws_.d_cub_temp_storage = nullptr;
    ws_.cub_temp_storage_bytes = 0;
    compact_doc_capacity_ = 0;
    compact_topk_capacity_ = 0;
    compact_doc_buffer_bytes_ = 0;
    size_t temp_bytes = 0;

    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr,
        temp_bytes,
        ws_.d_stage1_doc_scores,
        ws_.d_topk_scores,
        ws_.d_topk_doc_ids,
        ws_.d_sorted_doc_ids,
        static_cast<int>(std::max<size_t>(ws_.max_stage1_touched_docs, 1)),
        0,
        32);
    ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);

#ifndef GPU_MVR_OVERLAP_STAGE23
    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr,
        temp_bytes,
        ws_.d_doc_scores,
        ws_.d_topk_scores,
        ws_.d_selected_indices,
        ws_.d_topk_indices,
        static_cast<int>(ws_.max_stage2_candidates),
        0,
        32);
    ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);
#endif

    {
        thrust::device_ptr<int> len_ptr(ws_.d_pair_doc_ids);
        auto xform_iter = thrust::make_transform_iterator(len_ptr, cast_int_size_t_v8_base());
        temp_bytes = 0;
        cub::DeviceScan::InclusiveScan(
            nullptr,
            temp_bytes,
            xform_iter,
            ws_.d_candidate_offsets + 1,
            thrust::plus<size_t>(),
            static_cast<int>(ws_.max_stage2_candidates));
        ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);
    }

    temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_bytes,
        d_doc_bitmap_offsets_,
        d_doc_bitmap_offsets_,
        doc_bitmap_offset_count_);
    ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);

    CUDA_CHECK(cudaMalloc(&ws_.d_cub_temp_storage, ws_.cub_temp_storage_bytes));

    ensure_compact_doc_capacity(doc_buf_rows);

#ifndef GPU_MVR_OVERLAP_STAGE23
    CUDA_CHECK(cudaMalloc(&ws_.d_candidate_offsets, (ws_.max_stage2_candidates + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&ws_.d_token_dists, ws_.max_stage2_tokens * Q_DOCLEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_doc_scores, ws_.max_stage2_candidates * sizeof(float)));
#endif

    CUDA_CHECK(cudaMalloc(&ws_.d_selected_indices, ws_.max_stage2_k * sizeof(int)));

#ifdef GPU_MVR_USE_LUT
    CUDA_CHECK(cudaMalloc(&ws_.d_lut, LUT_TOTAL_FLOATS * sizeof(float)));
    const size_t stage2_lut_smem =
        STAGE2_LUT_SMEM_FLOATS * sizeof(float) + STAGE2_LUT_TILE_Q * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(
#ifdef GPU_MVR_STAGE2_DOC_LAYOUT
        stage2_binary_ip_lut_doc_kernel,
#else
        stage2_binary_ip_lut_kernel,
#endif
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        stage2_lut_smem));
#endif

    CUDA_CHECK(cudaMalloc(&ws_.d_cagra_dists, Q_DOCLEN * nprobe * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_cagra_labels, Q_DOCLEN * nprobe * sizeof(uint32_t)));

    CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_cb1_sumq, Q_DOCLEN * sizeof(float)));
#ifndef GPU_MVR_OVERLAP_STAGE23
    CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_batch_scores, ws_.max_stage2_candidates * sizeof(float)));
#endif

#ifdef GPU_MVR_OVERLAP_STAGE23
    CUDA_CHECK(cudaMalloc(&ws_.d_pst_candidate_offsets,
                          (ws_.max_stage2_candidates + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pst_clustered_pos,
                          ws_.max_stage2_tokens * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ws_.d_token_dists,
                          ws_.max_stage2_tokens * static_cast<size_t>(Q_DOCLEN) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_doc_scores, ws_.max_stage2_candidates * sizeof(float)));

    CUDA_CHECK(cudaMallocHost(&ws_.h_mapped_doc_scores,
                              ws_.max_stage2_candidates * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_pst_candidate_offsets,
                              (ws_.max_stage2_candidates + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_pst_candidate_doc_ids,
                              ws_.max_stage2_candidates * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_pst_total_tokens, sizeof(size_t)));

    CUDA_CHECK(cudaEventCreate(&ws_.phase_a_start_event));
    CUDA_CHECK(cudaEventCreate(&ws_.phase_a_gather_done_event));
    CUDA_CHECK(cudaEventCreate(&ws_.phase_a_prefix_done_event));
    CUDA_CHECK(cudaEventCreate(&ws_.phase_a_token_ids_done_event));
    CUDA_CHECK(cudaEventCreate(&ws_.phase_a_d2h_done_event));

    CUDA_CHECK(cudaEventCreateWithFlags(&ws_.pst_compute_done, cudaEventDisableTiming));
    for (int i = 0; i < gpu_mvr_index_v8_base::Workspace::PST_NUM_D2H_CHUNKS; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ws_.pst_d2h_chunk_done[i], cudaEventDisableTiming));
    }
#endif

    CUDA_CHECK(cudaMemset(ws_.d_doc_query_max, 0, doc_buf_rows * Q_DOCLEN * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_doc_bitmap_, 0,
                          doc_bitmap_bucket_count_ * sizeof(doc_bitmap_bucket_t)));
    CUDA_CHECK(cudaMemset(d_doc_bitmap_offsets_, 0,
                          doc_bitmap_offset_count_ * sizeof(doc_bitmap_offset_t)));

    int least_prio, greatest_prio;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio));
    CUDA_CHECK(cudaStreamCreateWithPriority(&ws_.stream_compute, cudaStreamDefault, least_prio));
    CUDA_CHECK(cudaStreamCreate(&ws_.stream_h2d));
    CUDA_CHECK(cudaStreamCreate(&ws_.stream_d2h));
    CUDA_CHECK(cudaEventCreate(&ws_.event_h2d_done));

#ifdef GPU_MVR_PROFILE
    CUDA_CHECK(cudaEventCreate(&ws_.event_start));
    CUDA_CHECK(cudaEventCreate(&ws_.event_end));
    CUDA_CHECK(cudaEventCreate(&ws_.event_stage1_start));
    CUDA_CHECK(cudaEventCreate(&ws_.event_stage1_end));
    CUDA_CHECK(cudaEventCreate(&ws_.event_stage2_start));
    CUDA_CHECK(cudaEventCreate(&ws_.event_stage2_end));

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

    CUDA_CHECK(cudaEventCreate(&ws_.s23_pst_kernel_start));
    CUDA_CHECK(cudaEventCreate(&ws_.s23_pst_kernel_end));

    for (int i = 0; i < gpu_mvr_index_v8_base::Workspace::MAX_XFER_RECORDS; ++i) {
        CUDA_CHECK(cudaEventCreate(&ws_.xfer_records[i].start));
        CUDA_CHECK(cudaEventCreate(&ws_.xfer_records[i].end));
    }
    ws_.xfer_count = 0;
#endif

    ws_.running_indices.resize(ws_.max_stage2_candidates);
    ws_.h_sel_indices.resize(ws_.max_stage2_k);
    ws_.refined_scores.resize(ws_.max_stage2_k);
    ws_.seen_doc_map.resize(num_docs, false);
}

std::vector<size_t> gpu_mvr_index::search(const float* queries, size_t k) {
    return search_impl<false>(queries, k);
}

std::vector<size_t> gpu_mvr_index::search_profiled(const float* queries, size_t k) {
    return search_impl<true>(queries, k, nullptr, true);
}

std::vector<size_t> gpu_mvr_index::search_profiled(
    const float* queries,
    size_t k,
    gpu_search_profile_v8* profile,
    bool print_profile) {
    return search_impl<true>(queries, k, profile, print_profile);
}

template <bool kProfile>
std::vector<size_t> gpu_mvr_index::search_impl(
    const float* queries,
    size_t k,
    gpu_search_profile_v8* profile,
    bool print_profile) {
    auto search_start = std::chrono::high_resolution_clock::now();
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        ws_.xfer_count = 0;
        CUDA_CHECK(cudaEventRecord(ws_.event_start, ws_.stream_compute));
    }
#endif

    for (size_t i = 0; i < Q_DOCLEN; ++i) {
        rotator_->rotate(&queries[i * d], &ws_.h_pinned_queries[i * PADDED_DIM]);
    }

    std::vector<query_object> query_objs(Q_DOCLEN);
    for (size_t i = 0; i < Q_DOCLEN; ++i) {
        query_objs[i] = query_object(&ws_.h_pinned_queries[i * PADDED_DIM], PADDED_DIM, ex_bits);
        ws_.h_pinned_cb1_sumq[i] = query_objs[i].cb1_sumq;
    }

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
        XFER_RECORD_END(ws_.stream_h2d,
                        Q_DOCLEN * PADDED_DIM * sizeof(float) + Q_DOCLEN * sizeof(float),
                        true);
    }

    CUDA_CHECK(cudaEventRecord(ws_.event_h2d_done, ws_.stream_h2d));
    CUDA_CHECK(cudaStreamWaitEvent(ws_.stream_compute, ws_.event_h2d_done));

#ifdef GPU_MVR_USE_LUT
    precompute_lut_kernel<<<Q_DOCLEN, 256, 0, ws_.stream_compute>>>(
        ws_.d_queries, ws_.d_lut);
    CUDA_CHECK(cudaGetLastError());
#endif

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.event_stage1_start, ws_.stream_compute));
    }
#endif

    int actual_k_stage1 = 0;
    rank_cluster_dists_gpu_impl<kProfile>(
        query_objs.data(),
        nprobe,
        k_rank_cluster,
        actual_k_stage1,
        ws_.stream_compute);

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.event_stage1_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventRecord(ws_.event_stage2_start, ws_.stream_compute));
    }
#endif

    std::vector<size_t> result;
    gpu_mvr_index_v8_base::rank_stage23_persistent_impl<kProfile>(
        actual_k_stage1,
        k,
        k_rank_all_tokens,
        query_objs.data(),
        result,
        print_profile);

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.event_stage2_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventRecord(ws_.event_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventSynchronize(ws_.event_end));
    }
#endif

    auto search_end = std::chrono::high_resolution_clock::now();
    if constexpr (kProfile) {
        const float search_wall_ms =
            std::chrono::duration<float, std::milli>(search_end - search_start).count();
#ifdef GPU_MVR_PROFILE
        gpu_search_profile_v8 local_profile;
        gpu_search_profile_v8& out_profile = profile ? *profile : local_profile;
        out_profile = {};
        out_profile.search_wall_ms = search_wall_ms;

        float total_time = 0.0f;
        float stage1_time = 0.0f;
        float stage2_time = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_time, ws_.event_start, ws_.event_end));
        CUDA_CHECK(cudaEventElapsedTime(&stage1_time, ws_.event_stage1_start, ws_.event_stage1_end));
        CUDA_CHECK(cudaEventElapsedTime(&stage2_time, ws_.event_stage2_start, ws_.event_stage2_end));
        out_profile.total_search_time_ms = total_time;
        out_profile.stage1_time_ms = stage1_time;
        out_profile.stage2_time_ms = stage2_time;

        float s1_cagra = 0.0f;
        float s1_expansion = 0.0f;
        float s1_binary_ip = 0.0f;
        float s1_memset = 0.0f;
        float s1_atomic_agg = 0.0f;
        float s1_sum_scores = 0.0f;
        float s1_topk_sort = 0.0f;
        float s1_d2d = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&s1_cagra, ws_.s1_cagra_start, ws_.s1_cagra_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_expansion, ws_.s1_expansion_start, ws_.s1_expansion_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_binary_ip, ws_.s1_binary_ip_start, ws_.s1_binary_ip_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_memset, ws_.s1_memset_start, ws_.s1_memset_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_atomic_agg, ws_.s1_atomic_agg_start, ws_.s1_atomic_agg_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_sum_scores, ws_.s1_sum_scores_start, ws_.s1_sum_scores_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_topk_sort, ws_.s1_topk_sort_start, ws_.s1_topk_sort_end));
        CUDA_CHECK(cudaEventElapsedTime(&s1_d2d, ws_.s1_d2d_start, ws_.s1_d2d_end));
        out_profile.s1_cagra_ms = s1_cagra;
        out_profile.s1_expansion_ms = s1_expansion;
        out_profile.s1_binary_ip_ms = s1_binary_ip;
        out_profile.s1_atomic_agg_ms = s1_atomic_agg;
        out_profile.s1_sum_scores_ms = s1_sum_scores;
        out_profile.s1_topk_sort_ms = s1_topk_sort;
        out_profile.s1_d2d_ms = s1_d2d;
        out_profile.s1_memset_ms = s1_memset;
        out_profile.s1_sum_accounted_ms =
            s1_cagra + s1_expansion + s1_binary_ip + s1_atomic_agg +
            s1_sum_scores + s1_topk_sort + s1_d2d;

        const auto& stage23 = ws_.last_stage23_profile;
        out_profile.phase_a_wall_ms = stage23.phase_a_wall_ms;
        out_profile.phase_a_gpu_total_ms = stage23.phase_a_gpu_total_ms;
        out_profile.phase_a_gather_ms = stage23.phase_a_gather_ms;
        out_profile.phase_a_prefix_ms = stage23.phase_a_prefix_ms;
        out_profile.phase_a_token_ids_ms = stage23.phase_a_token_ids_ms;
        out_profile.phase_a_d2h_ms = stage23.phase_a_d2h_ms;
        out_profile.phase_a_cpu_launch_gather_ms = stage23.phase_a_cpu_launch_gather_ms;
        out_profile.phase_a_cpu_launch_prefix_ms = stage23.phase_a_cpu_launch_prefix_ms;
        out_profile.phase_a_cpu_launch_token_ids_ms = stage23.phase_a_cpu_launch_token_ids_ms;
        out_profile.phase_a_cpu_launch_d2h_ms = stage23.phase_a_cpu_launch_d2h_ms;
        out_profile.phase_a_cpu_sync_ms = stage23.phase_a_cpu_sync_ms;
        out_profile.phase_a_cpu_event_elapsed_ms = stage23.phase_a_cpu_event_elapsed_ms;
        out_profile.phase_a_gpu_sum_accounted_ms = stage23.phase_a_gpu_sum_accounted_ms;
        out_profile.phase_b_wall_ms = stage23.phase_b_wall_ms;
        out_profile.phase_b_binary_ip_total_ms = stage23.phase_b_binary_ip_total_ms;
        out_profile.phase_b_doc_score_total_ms = stage23.phase_b_doc_score_total_ms;
        out_profile.phase_b_total_kernel_ms = stage23.phase_b_total_kernel_ms;
        out_profile.phase_c_wait_d2h_ms = stage23.phase_c_wait_d2h_ms;
        out_profile.phase_c_topk_ms = stage23.phase_c_topk_ms;
        out_profile.phase_c_identify_ms = stage23.phase_c_identify_ms;
        out_profile.phase_c_prepare_ms = stage23.phase_c_prepare_ms;
        out_profile.phase_c_cpu_refine_ms = stage23.phase_c_cpu_refine_ms;
        out_profile.phase_c_combine_ms = stage23.phase_c_combine_ms;
        out_profile.phase_c_total_ms = stage23.phase_c_total_ms;
        out_profile.phase_c_refined_docs = stage23.phase_c_refined_docs;
        out_profile.phase_abc_total_wall_ms = stage23.phase_abc_total_wall_ms;

        for (int xi = 0; xi < ws_.xfer_count; ++xi) {
            float xfer_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(
                &xfer_ms,
                ws_.xfer_records[xi].start,
                ws_.xfer_records[xi].end));
            out_profile.transfer_total_ms += xfer_ms;
            out_profile.transfer_count += 1.0;
            if (ws_.xfer_records[xi].is_h2d) {
                out_profile.transfer_h2d_ms += xfer_ms;
                out_profile.transfer_h2d_bytes += ws_.xfer_records[xi].bytes;
            } else {
                out_profile.transfer_d2h_ms += xfer_ms;
                out_profile.transfer_d2h_bytes += ws_.xfer_records[xi].bytes;
            }
        }

        if (print_profile) {
            print_gpu_search_profile_v8(out_profile, "[PROFILE]");
        }
#else
        if (print_profile) {
            std::cout << "[SEARCH] Total wall-clock time: " << search_wall_ms << " ms\n";
        }
#endif
    }

    return result;
}

void gpu_mvr_index::rank_cluster_dists_gpu(
    query_object* h_query_objs,
    size_t nprobe_value,
    size_t k,
    int& actual_k_out,
    cudaStream_t stream) {
    rank_cluster_dists_gpu_impl<false>(h_query_objs, nprobe_value, k, actual_k_out, stream);
}

template <bool kProfile>
void gpu_mvr_index::rank_cluster_dists_gpu_impl(
    query_object* h_query_objs,
    size_t nprobe_value,
    size_t k,
    int& actual_k_out,
    cudaStream_t stream) {
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_memset_start, ws_.stream_d2h));
    }
#endif
    CUDA_CHECK(cudaMemsetAsync(d_doc_bitmap_, 0,
                               doc_bitmap_bucket_count_ * sizeof(doc_bitmap_bucket_t),
                               ws_.stream_d2h));
    CUDA_CHECK(cudaMemsetAsync(d_doc_bitmap_offsets_, 0,
                               doc_bitmap_offset_count_ * sizeof(doc_bitmap_offset_t),
                               ws_.stream_d2h));
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_memset_end, ws_.stream_d2h));
    }
#endif
    CUDA_CHECK(cudaEventRecord(ws_.event_h2d_done, ws_.stream_d2h));

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_cagra_start, stream));
    }
#endif
    ivf->search_batch_gpu(
        cagra_res_,
        ws_.d_queries,
        Q_DOCLEN,
        nprobe_value,
        ws_.d_cagra_dists,
        ws_.d_cagra_labels,
        stream,
        static_cast<size_t>(itopk_size));
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_cagra_end, stream));
        CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_start, stream));
        CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_end, stream));
    }
#endif

    CUDA_CHECK(cudaStreamWaitEvent(stream, ws_.event_h2d_done));

    const int threads_per_block = 256;
    const int blocks_x = std::max(
        (max_cluster_size + threads_per_block - 1) / threads_per_block,
        16);
    const dim3 grid(blocks_x, Q_DOCLEN);

    stage1_binary_ip_lut_flag_docs_v8_kernel<<<grid, threads_per_block, 0, stream>>>(
        ws_.d_cagra_labels,
        d_cluster_pos_,
        d_clustered_doc_ids_,
        d_doc_bitmap_,
        num_docs,
        static_cast<int>(nprobe_value),
        ivf->n_clusters);
    CUDA_CHECK(cudaGetLastError());

    bitmap_offset_init_v8_kernel<<<
        (doc_bitmap_bucket_count_ + threads_per_block - 1) / threads_per_block,
        threads_per_block,
        0,
        stream>>>(
            d_doc_bitmap_,
            doc_bitmap_bucket_count_,
            d_doc_bitmap_offsets_);
    CUDA_CHECK(cudaGetLastError());

    size_t scan_temp = ws_.cub_temp_storage_bytes;
    cub::DeviceScan::ExclusiveSum(
        ws_.d_cub_temp_storage,
        scan_temp,
        d_doc_bitmap_offsets_,
        d_doc_bitmap_offsets_,
        doc_bitmap_offset_count_,
        stream);

    doc_bitmap_offset_t h_num_touched_u32 = 0;
    if constexpr (kProfile) {
        XFER_RECORD_BEGIN(stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(&h_num_touched_u32,
                               d_doc_bitmap_offsets_ + doc_bitmap_offset_count_ - 1,
                               sizeof(doc_bitmap_offset_t),
                               cudaMemcpyDeviceToHost,
                               stream));
    if constexpr (kProfile) {
        XFER_RECORD_END(stream, sizeof(doc_bitmap_offset_t), false);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int h_num_touched = static_cast<int>(h_num_touched_u32);
    if (h_num_touched == 0) {
        actual_k_out = 0;
        return;
    }
    if (static_cast<size_t>(h_num_touched) > max_compact_docs_) {
        throw std::runtime_error(
            "v8 compact-doc count " + std::to_string(h_num_touched) +
            " exceeded workspace bound " + std::to_string(max_compact_docs_));
    }

    ensure_compact_doc_capacity(static_cast<size_t>(h_num_touched));
    CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_query_max,
                               0,
                               static_cast<size_t>(h_num_touched) * Q_DOCLEN * sizeof(float),
                               stream));

    bitmap_unique_docs_v8_kernel<<<
        (doc_bitmap_bucket_count_ + threads_per_block - 1) / threads_per_block,
        threads_per_block,
        0,
        stream>>>(
            d_doc_bitmap_,
            doc_bitmap_bucket_count_,
            d_doc_bitmap_offsets_,
            ws_.d_unique_doc_ids);
    CUDA_CHECK(cudaGetLastError());

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_start, stream));
    }
#endif
#ifdef GPU_MVR_USE_LUT
    stage1_binary_ip_lut_v8_kernel<<<grid, threads_per_block, 0, stream>>>(
        ws_.d_lut,
        d_clustered_code_,
        d_clustered_factor_,
        ws_.d_cb1_sumq,
        ws_.d_cagra_labels,
        d_cluster_pos_,
        d_clustered_doc_ids_,
        ws_.d_doc_query_max,
        d_doc_bitmap_,
        d_doc_bitmap_offsets_,
        h_num_touched,
        num_docs,
        static_cast<int>(nprobe_value),
        ivf->n_clusters);
#else
    stage1_binary_ip_nolut_v8_kernel<<<grid, threads_per_block, 0, stream>>>(
        ws_.d_queries,
        d_clustered_code_,
        d_clustered_factor_,
        ws_.d_cb1_sumq,
        ws_.d_cagra_labels,
        d_cluster_pos_,
        d_clustered_doc_ids_,
        ws_.d_doc_query_max,
        d_doc_bitmap_,
        d_doc_bitmap_offsets_,
        h_num_touched,
        num_docs,
        static_cast<int>(nprobe_value),
        ivf->n_clusters);
#endif
    CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_end, stream));
        CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_start, stream));
        CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_end, stream));
    }
#endif

    actual_k_out = std::min(static_cast<int>(k), h_num_touched);

    const int thread_count = SUM_SCORES_WARPS_PER_BLOCK * 32;
    const int sparse_blocks =
        (h_num_touched + SUM_SCORES_WARPS_PER_BLOCK - 1) / SUM_SCORES_WARPS_PER_BLOCK;
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_sum_scores_start, stream));
    }
#endif
    sum_doc_scores_compact_v8_kernel<<<sparse_blocks, thread_count, 0, stream>>>(
        ws_.d_doc_query_max,
        ws_.d_unique_doc_ids,
        ws_.d_stage1_doc_scores,
        ws_.d_topk_doc_ids,
        h_num_touched);
    CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_sum_scores_end, stream));
        CUDA_CHECK(cudaEventRecord(ws_.s1_topk_sort_start, stream));
    }
#endif
    cub::DeviceRadixSort::SortPairsDescending(
        ws_.d_cub_temp_storage,
        ws_.cub_temp_storage_bytes,
        ws_.d_stage1_doc_scores,
        ws_.d_topk_scores,
        ws_.d_topk_doc_ids,
        ws_.d_sorted_doc_ids,
        h_num_touched,
        0,
        32,
        stream);
    CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_topk_sort_end, stream));
        CUDA_CHECK(cudaEventRecord(ws_.s1_d2d_start, stream));
    }
#endif
    CUDA_CHECK(cudaMemcpyAsync(ws_.d_topk_doc_ids,
                               ws_.d_sorted_doc_ids,
                               actual_k_out * sizeof(int),
                               cudaMemcpyDeviceToDevice,
                               stream));
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_d2d_end, stream));
    }
#endif
}
