#include "gpu_index_v7.cuh"

#define CHIMERA_V3_SKIP_WORKSPACE_ALLOC 1
#define chimera_index chimera_index_v7_base
#define cast_int_size_t cast_int_size_t_v7_base
#include "gpu_index_v3.cu"
#undef cast_int_size_t
#undef chimera_index
#undef CHIMERA_V3_SKIP_WORKSPACE_ALLOC

#include <cub/cub.cuh>

namespace Chimera {

namespace {

constexpr size_t kV7CompactDocRowAlignment = 1u << 16;  // 65536 rows
constexpr size_t kV7CompactDocMinSlackRows = 1u << 15;  // 32768 rows
constexpr size_t kV7CompactDocBufferAlignment = 128;

size_t v7_sum_top_sorted_counts(std::vector<size_t>& counts, size_t top_n) {
    if (counts.empty() || top_n == 0) {
        return 0;
    }

    std::sort(counts.begin(), counts.end(), std::greater<size_t>());
    top_n = std::min(top_n, counts.size());
    return std::accumulate(counts.begin(), counts.begin() + top_n, size_t{0});
}

std::pair<size_t, size_t> v7_compute_stage2_token_bounds(
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

size_t v7_align_up(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

char* v7_align_up_ptr(char* ptr, size_t alignment) {
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    const auto aligned = v7_align_up(addr, alignment);
    return reinterpret_cast<char*>(aligned);
}

size_t v7_initial_compact_doc_capacity(
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
    estimate = std::min(max_compact_docs, v7_align_up(estimate, kV7CompactDocRowAlignment));
    return std::max<size_t>(estimate, 1);
}

size_t v7_grow_compact_doc_capacity(size_t current, size_t required, size_t limit) {
    const size_t aligned_required =
        v7_align_up(std::max(required, size_t{1}), kV7CompactDocRowAlignment);
    const size_t required_slack =
        std::max(required / 16, kV7CompactDocMinSlackRows);
    const size_t current_slack =
        std::max(current / 8, kV7CompactDocMinSlackRows);
    size_t target = aligned_required;
    target = std::max(
        target,
        v7_align_up(required + required_slack, kV7CompactDocRowAlignment));
    if (current > 0) {
        target = std::max(
            target,
            v7_align_up(current + current_slack, kV7CompactDocRowAlignment));
    }
    return std::min(limit, target);
}

size_t v7_compact_doc_arena_bytes(size_t row_capacity, size_t topk_capacity) {
    size_t offset = 0;
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    offset += row_capacity * sizeof(int);
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    offset += row_capacity * sizeof(int);
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    offset += row_capacity * sizeof(float);
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    offset += row_capacity * Q_DOCLEN * sizeof(float);
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    offset += topk_capacity * sizeof(float);
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    offset += topk_capacity * sizeof(int);
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    offset += topk_capacity * sizeof(int);
    return offset;
}

void v7_bind_compact_doc_arena(
    chimera_index_v7_base::Workspace& ws,
    char* base,
    size_t row_capacity,
    size_t topk_capacity) {
    size_t offset = 0;
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    ws.d_sorted_doc_ids = reinterpret_cast<int*>(base + offset);
    offset += row_capacity * sizeof(int);
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    ws.d_unique_doc_ids = reinterpret_cast<int*>(base + offset);
    offset += row_capacity * sizeof(int);
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    ws.d_stage1_doc_scores = reinterpret_cast<float*>(base + offset);
    offset += row_capacity * sizeof(float);
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    ws.d_doc_query_max = reinterpret_cast<float*>(base + offset);
    offset += row_capacity * Q_DOCLEN * sizeof(float);
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    ws.d_topk_scores = reinterpret_cast<float*>(base + offset);
    offset += topk_capacity * sizeof(float);
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    ws.d_topk_doc_ids = reinterpret_cast<int*>(base + offset);
    offset += topk_capacity * sizeof(int);
    offset = v7_align_up(offset, kV7CompactDocBufferAlignment);
    ws.d_topk_indices = reinterpret_cast<int*>(base + offset);
}

int v7_doc_id_from_doc_ptrs_host_binary_range(
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

int v7_doc_id_from_doc_ptrs_host(
    const std::vector<int>& doc_ptrs,
    uint32_t token_id) {
    return v7_doc_id_from_doc_ptrs_host_binary_range(
        doc_ptrs,
        0,
        static_cast<int>(doc_ptrs.size()) - 2,
        token_id);
}

}  // namespace

chimera_index::chimera_index(
    const std::string& filename,
    const std::vector<int>& doc_lens,
    const gpu_search_runtime_options& runtime_options)
    : chimera_index_v7_base(filename, doc_lens, runtime_options) {
    const char* stage1_mode_env = std::getenv("CHIMERA_V7_STAGE1_MODE");
    if (stage1_mode_env != nullptr) {
        const std::string mode(stage1_mode_env);
        if (mode == "memory_padded_align" || mode == "padded") {
            stage1_align_mode_ = Stage1AlignMode::MemoryPaddedAlign;
        } else if (mode == "compact_align" || mode == "compact") {
            stage1_align_mode_ = Stage1AlignMode::CompactAlign;
        } else {
            throw std::runtime_error(
                "Unknown CHIMERA_V7_STAGE1_MODE=" + mode +
                " (expected compact_align or memory_padded_align)");
        }
    }

    maybe_load_stage1_padded_layout(filename);
    compute_workspace_probe_bounds();
    allocate_workspace();
}

chimera_index::~chimera_index() {
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_doc_bitmap_));
    CUDA_CHECK(cudaFree(d_doc_bitmap_offsets_));
    CUDA_CHECK(cudaFree(d_compact_doc_buffer_raw_));
    CUDA_CHECK(cudaFree(d_stage1_padded_code_));
    CUDA_CHECK(cudaFree(d_stage1_padded_factor_));
    CUDA_CHECK(cudaFree(d_stage1_padded_doc_ids_));
    CUDA_CHECK(cudaFree(d_stage1_padded_cluster_meta_));
    d_doc_bitmap_ = nullptr;
    d_doc_bitmap_offsets_ = nullptr;
    d_compact_doc_buffer_raw_ = nullptr;
    d_compact_doc_buffer_ = nullptr;
    d_stage1_padded_code_ = nullptr;
    d_stage1_padded_factor_ = nullptr;
    d_stage1_padded_doc_ids_ = nullptr;
    d_stage1_padded_cluster_meta_ = nullptr;

    ws_.d_sorted_doc_ids = nullptr;
    ws_.d_unique_doc_ids = nullptr;
    ws_.d_stage1_doc_scores = nullptr;
    ws_.d_doc_query_max = nullptr;
    ws_.d_topk_scores = nullptr;
    ws_.d_topk_doc_ids = nullptr;
    ws_.d_topk_indices = nullptr;
    ws_.d_doc_touched = nullptr;
}

void chimera_index::maybe_load_stage1_padded_layout(const std::string& filename) {
    if (stage1_align_mode_ != Stage1AlignMode::MemoryPaddedAlign) {
        std::cout << "[v7] Stage1 align mode: compact_align\n";
        return;
    }

    const auto resolved_paths = gpu_index_layout::resolve_index_paths(filename);
    const auto& gpu_index_path = resolved_paths.gpu_index_path;
    if (gpu_index_path.empty() || !std::filesystem::exists(gpu_index_path)) {
        throw std::runtime_error(
            "Missing cluster_1bit.bin for v7 padded stage1 layout in " +
            std::filesystem::path(resolved_paths.quantized_data_path).parent_path().string());
    }

    const size_t inv_n = ivf->inv_list.size();
    const size_t code_per_vec = PADDED_DIM / 8;

    std::ifstream clustered_header_file(gpu_index_path, std::ios::binary);
    const auto clustered_header =
        clustered_format::read_header(
            clustered_header_file, gpu_index_path);
    if (clustered_header.n_entries != inv_n ||
        clustered_header.code_bytes_per_vector != code_per_vec) {
        throw std::runtime_error(
            "GPU index metadata mismatch for " + gpu_index_path);
    }
    clustered_header_file.close();

    struct stat clustered_stat {};
    const int clustered_fd = open(gpu_index_path.c_str(), O_RDONLY);
    if (clustered_fd < 0) {
        throw std::runtime_error("Failed to open GPU index: " + gpu_index_path);
    }
    if (fstat(clustered_fd, &clustered_stat) != 0) {
        const auto err = errno;
        close(clustered_fd);
        throw std::runtime_error(
            "Failed to stat GPU index " + gpu_index_path + ": " +
            std::system_category().message(err));
    }

    const size_t mapping_size = static_cast<size_t>(clustered_stat.st_size);
    void* clustered_mapping =
        mmap(nullptr, mapping_size, PROT_READ, MAP_PRIVATE, clustered_fd, 0);
    if (clustered_mapping == MAP_FAILED) {
        const auto err = errno;
        close(clustered_fd);
        throw std::runtime_error(
            "Failed to mmap GPU index " + gpu_index_path + ": " +
            std::system_category().message(err));
    }

    const auto* clustered_base = static_cast<const char*>(clustered_mapping);
    const size_t clustered_prefix_bytes =
        clustered_format::prefix_bytes(clustered_header);
    const auto* clustered_code = clustered_base + clustered_prefix_bytes;
    const auto* clustered_factor = reinterpret_cast<const float*>(
        clustered_code + inv_n * code_per_vec);
    const auto* clustered_doc_ids = reinterpret_cast<const int*>(
        reinterpret_cast<const char*>(clustered_factor) + inv_n * sizeof(float));

    std::vector<uint2> host_cluster_meta(n_clusters);
    stage1_padded_tokens_ = 0;
    for (size_t cluster_id = 0; cluster_id < n_clusters; ++cluster_id) {
        stage1_padded_tokens_ =
            v7_align_up(stage1_padded_tokens_, kStage1ClusterAlignTokens);
        const size_t start = ivf->cluster_pos[cluster_id];
        const size_t end = ivf->cluster_pos[cluster_id + 1];
        const size_t cluster_size = end - start;
        if (stage1_padded_tokens_ > std::numeric_limits<uint32_t>::max() ||
            cluster_size > std::numeric_limits<uint32_t>::max()) {
            munmap(clustered_mapping, mapping_size);
            close(clustered_fd);
            throw std::runtime_error(
                "v7 padded stage1 layout exceeded uint32 addressable range");
        }
        host_cluster_meta[cluster_id] = make_uint2(
            static_cast<uint32_t>(stage1_padded_tokens_),
            static_cast<uint32_t>(cluster_size));
        stage1_padded_tokens_ += cluster_size;
    }

    std::vector<char> host_padded_code(stage1_padded_tokens_ * code_per_vec, 0);
    std::vector<float> host_padded_factor(stage1_padded_tokens_, 0.0f);
    std::vector<int> host_padded_doc_ids(stage1_padded_tokens_, -1);

    for (size_t cluster_id = 0; cluster_id < n_clusters; ++cluster_id) {
        const size_t src_start = ivf->cluster_pos[cluster_id];
        const size_t src_end = ivf->cluster_pos[cluster_id + 1];
        const size_t cluster_size = src_end - src_start;
        if (cluster_size == 0) {
            continue;
        }

        const size_t dst_start = host_cluster_meta[cluster_id].x;
        std::memcpy(
            host_padded_code.data() + dst_start * code_per_vec,
            clustered_code + src_start * code_per_vec,
            cluster_size * code_per_vec);
        std::memcpy(
            host_padded_factor.data() + dst_start,
            clustered_factor + src_start,
            cluster_size * sizeof(float));
        std::memcpy(
            host_padded_doc_ids.data() + dst_start,
            clustered_doc_ids + src_start,
            cluster_size * sizeof(int));
    }

    CUDA_CHECK(cudaMalloc(&d_stage1_padded_code_, stage1_padded_tokens_ * code_per_vec));
    CUDA_CHECK(cudaMalloc(&d_stage1_padded_factor_, stage1_padded_tokens_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stage1_padded_doc_ids_, stage1_padded_tokens_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_stage1_padded_cluster_meta_, n_clusters * sizeof(uint2)));

    CUDA_CHECK(cudaMemcpy(
        d_stage1_padded_code_,
        host_padded_code.data(),
        stage1_padded_tokens_ * code_per_vec,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_stage1_padded_factor_,
        host_padded_factor.data(),
        stage1_padded_tokens_ * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_stage1_padded_doc_ids_,
        host_padded_doc_ids.data(),
        stage1_padded_tokens_ * sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_stage1_padded_cluster_meta_,
        host_cluster_meta.data(),
        n_clusters * sizeof(uint2),
        cudaMemcpyHostToDevice));

    munmap(clustered_mapping, mapping_size);
    close(clustered_fd);

    const size_t padded_bytes =
        stage1_padded_tokens_ * (code_per_vec + sizeof(float) + sizeof(int)) +
        n_clusters * sizeof(uint2);
    std::cout << "[v7] Stage1 align mode: memory_padded_align ("
              << (padded_bytes / (1024.0 * 1024.0)) << " MB, padded_tokens="
              << stage1_padded_tokens_ << ")\n";
}

void chimera_index::compute_workspace_probe_bounds() {
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
        int prev_doc_id = v7_doc_id_from_doc_ptrs_host(doc_ptrs_, first_token_id);
        size_t unique_doc_count = 1;

        for (size_t pos = start + 1; pos < end; ++pos) {
            const uint32_t token_id = static_cast<uint32_t>(ivf->inv_list[pos]);
            const int current_doc_id = v7_doc_id_from_doc_ptrs_host_binary_range(
                doc_ptrs_, prev_doc_id, static_cast<int>(num_docs) - 1, token_id);
            if (current_doc_id != prev_doc_id) {
                ++unique_doc_count;
                prev_doc_id = current_doc_id;
            }
        }

        cluster_unique_doc_counts[cluster_id] = unique_doc_count;
    }

    workspace_probe_token_bound_ =
        v7_sum_top_sorted_counts(cluster_token_counts, probe_cluster_bound);
    workspace_probe_unique_doc_bound_ =
        v7_sum_top_sorted_counts(cluster_unique_doc_counts, probe_cluster_bound);
}

void chimera_index::ensure_compact_doc_capacity(size_t required_rows) {
    required_rows = std::max<size_t>(required_rows, 1);
    if (required_rows <= compact_doc_capacity_ &&
        d_compact_doc_buffer_raw_ != nullptr) {
        return;
    }

    const size_t grown_rows = v7_grow_compact_doc_capacity(
        compact_doc_capacity_, required_rows, static_cast<size_t>(num_docs));
    const size_t required_topk_capacity =
        std::max(grown_rows, ws_.max_stage2_candidates);
    const size_t required_buffer_bytes =
        v7_compact_doc_arena_bytes(grown_rows, required_topk_capacity);

    std::cout << "[workspace][v7] Growing compact doc capacity from "
              << compact_doc_capacity_ << " to " << grown_rows
              << " rows (required=" << required_rows << ")" << std::endl;

    CUDA_CHECK(cudaFree(d_compact_doc_buffer_raw_));
    d_compact_doc_buffer_raw_ = nullptr;
    d_compact_doc_buffer_ = nullptr;
    compact_doc_buffer_bytes_ = required_buffer_bytes;
    CUDA_CHECK(cudaMalloc(
        &d_compact_doc_buffer_raw_,
        compact_doc_buffer_bytes_ + kV7CompactDocBufferAlignment - 1));
    d_compact_doc_buffer_ =
        v7_align_up_ptr(d_compact_doc_buffer_raw_, kV7CompactDocBufferAlignment);
    v7_bind_compact_doc_arena(ws_, d_compact_doc_buffer_, grown_rows, required_topk_capacity);

    compact_doc_capacity_ = grown_rows;
    compact_topk_capacity_ = required_topk_capacity;
}

void chimera_index::allocate_workspace() {
    ws_.max_q_doclen = Q_DOCLEN;
    ws_.max_stage1_pairs = workspace_probe_token_bound_ * Q_DOCLEN;
    ws_.max_stage2_candidates = k_rank_cluster;
    ws_.max_stage2_k = k_rank_all_tokens;
    std::tie(ws_.max_stage2_tokens, ws_.max_stage2_k_tokens) =
        v7_compute_stage2_token_bounds(doc_ptrs_, ws_.max_stage2_candidates, ws_.max_stage2_k);
    ws_.estimated_num_docs = static_cast<size_t>(num_docs);
    ws_.max_stage1_touched_docs =
        std::min(ws_.estimated_num_docs, ws_.max_stage1_pairs);

    max_compact_docs_ = std::min(
        static_cast<size_t>(num_docs),
        workspace_probe_unique_doc_bound_ * Q_DOCLEN);
    doc_bitmap_bucket_count_ = doc_bitmap_num_buckets(num_docs);
    doc_bitmap_offset_count_ = doc_bitmap_num_offsets(doc_bitmap_bucket_count_);
    const size_t initial_compact_doc_capacity = v7_initial_compact_doc_capacity(
        n,
        n_clusters,
        workspace_probe_cluster_bound_,
        max_compact_docs_);
    const size_t doc_buf_rows = initial_compact_doc_capacity;

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
    std::cout << "workspace_stage2_candidate_token_bound: "
              << ws_.max_stage2_tokens << std::endl;
    std::cout << "workspace_stage2_topk_token_bound: "
              << ws_.max_stage2_k_tokens << std::endl;

#ifdef CHIMERA_USE_LUT
    const size_t stage1_emb_ids_bytes = 0;
    const size_t stage1_pair_offsets_bytes = 0;
    const size_t stage1_emb_dists_bytes = 0;
    const size_t phase_a_doc_lengths_bytes =
        ws_.max_stage2_candidates * sizeof(int);
#else
    const size_t stage1_emb_ids_bytes =
        ws_.max_stage1_pairs * sizeof(size_t);
    const size_t stage1_pair_offsets_bytes =
        (Q_DOCLEN + 1) * sizeof(int);
    const size_t stage1_emb_dists_bytes =
        ws_.max_stage1_pairs * sizeof(float);
    const size_t phase_a_doc_lengths_bytes =
        ws_.max_stage1_pairs * sizeof(int);
#endif

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
    std::cout << "Initial GPU memory required for v7 workspace: "
              << (estimated_size / (1024.0 * 1024.0)) << " MB" << std::endl;

    CUDA_CHECK(cudaMalloc(&ws_.d_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_cb1_sumq, Q_DOCLEN * sizeof(float)));
#ifndef CHIMERA_USE_LUT
    CUDA_CHECK(cudaMalloc(&ws_.d_emb_ids, ws_.max_stage1_pairs * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_offsets, (Q_DOCLEN + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_emb_dists, ws_.max_stage1_pairs * sizeof(float)));
#endif
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

#ifndef CHIMERA_OVERLAP_STAGE23
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

#ifndef CHIMERA_USE_LUT
    temp_bytes = 0;
    cub::DeviceScan::InclusiveScan(
        nullptr,
        temp_bytes,
        ws_.d_pair_offsets + 1,
        ws_.d_pair_offsets + 1,
        thrust::plus<size_t>(),
        static_cast<int>(Q_DOCLEN));
    ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);
#endif

    {
        thrust::device_ptr<int> len_ptr(ws_.d_pair_doc_ids);
        auto xform_iter = thrust::make_transform_iterator(len_ptr, cast_int_size_t_v7_base());
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

#ifndef CHIMERA_OVERLAP_STAGE23
    CUDA_CHECK(cudaMalloc(&ws_.d_candidate_offsets, (ws_.max_stage2_candidates + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&ws_.d_token_dists, ws_.max_stage2_tokens * Q_DOCLEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_doc_scores, ws_.max_stage2_candidates * sizeof(float)));
#endif

    CUDA_CHECK(cudaMalloc(&ws_.d_selected_indices, ws_.max_stage2_k * sizeof(int)));

#ifdef CHIMERA_USE_LUT
    CUDA_CHECK(cudaMalloc(&ws_.d_lut, LUT_TOTAL_FLOATS * sizeof(float)));
    const size_t stage2_lut_smem =
        STAGE2_LUT_SMEM_FLOATS * sizeof(float) + STAGE2_LUT_TILE_Q * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(
        stage2_binary_ip_lut_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        stage2_lut_smem));
#endif

    CUDA_CHECK(cudaMalloc(&ws_.d_cagra_dists, Q_DOCLEN * nprobe * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_cagra_labels, Q_DOCLEN * nprobe * sizeof(uint32_t)));

    CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_cb1_sumq, Q_DOCLEN * sizeof(float)));
#ifndef CHIMERA_OVERLAP_STAGE23
    CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_batch_scores, ws_.max_stage2_candidates * sizeof(float)));
#endif

#ifdef CHIMERA_OVERLAP_STAGE23
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
    for (int i = 0; i < chimera_index_v7_base::Workspace::PST_NUM_D2H_CHUNKS; ++i) {
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

#ifdef CHIMERA_PROFILE
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

    for (int i = 0; i < chimera_index_v7_base::Workspace::MAX_XFER_RECORDS; ++i) {
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

std::vector<size_t> chimera_index::search(const float* queries, size_t k) {
    return search_impl<false>(queries, k);
}

std::vector<size_t> chimera_index::search_profiled(const float* queries, size_t k) {
    return search_impl<true>(queries, k);
}

template <bool kProfile>
std::vector<size_t> chimera_index::search_impl(const float* queries, size_t k) {
    auto search_start = std::chrono::high_resolution_clock::now();
#ifdef CHIMERA_PROFILE
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

#ifdef CHIMERA_USE_LUT
    precompute_lut_kernel<<<Q_DOCLEN, 256, 0, ws_.stream_compute>>>(
        ws_.d_queries, ws_.d_lut);
    CUDA_CHECK(cudaGetLastError());
#endif

#ifdef CHIMERA_PROFILE
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

#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.event_stage1_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventRecord(ws_.event_stage2_start, ws_.stream_compute));
    }
#endif

    std::vector<size_t> result;
#ifdef CHIMERA_OVERLAP_STAGE23
    chimera_index_v7_base::rank_stage23_persistent_impl<kProfile>(
        actual_k_stage1,
        k,
        k_rank_all_tokens,
        query_objs.data(),
        result);
#else
#error "gpu_search_v7 requires CHIMERA_OVERLAP_STAGE23"
#endif

#ifdef CHIMERA_PROFILE
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
        std::cout << "[SEARCH] Total wall-clock time: " << search_wall_ms << " ms\n";
    }

    return result;
}

void chimera_index::rank_cluster_dists_gpu(
    query_object* h_query_objs,
    size_t nprobe_value,
    size_t k,
    int& actual_k_out,
    cudaStream_t stream) {
    rank_cluster_dists_gpu_impl<false>(h_query_objs, nprobe_value, k, actual_k_out, stream);
}

template <bool kProfile>
void chimera_index::rank_cluster_dists_gpu_impl(
    query_object* h_query_objs,
    size_t nprobe_value,
    size_t k,
    int& actual_k_out,
    cudaStream_t stream) {
#ifdef CHIMERA_PROFILE
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
#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_memset_end, ws_.stream_d2h));
    }
#endif
    CUDA_CHECK(cudaEventRecord(ws_.event_h2d_done, ws_.stream_d2h));

#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_cagra_start, stream));
    }
#endif
    ivf->search_batch_gpu(
        ws_.d_queries,
        Q_DOCLEN,
        nprobe_value,
        ws_.d_cagra_dists,
        ws_.d_cagra_labels,
        stream,
        static_cast<size_t>(itopk_size));
#ifdef CHIMERA_PROFILE
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

    if (stage1_align_mode_ == Stage1AlignMode::MemoryPaddedAlign) {
        stage1_binary_ip_lut_flag_docs_v7_padded_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_cagra_labels,
            d_stage1_padded_cluster_meta_,
            d_stage1_padded_doc_ids_,
            d_doc_bitmap_,
            num_docs,
            static_cast<int>(nprobe_value),
            ivf->n_clusters);
        CUDA_CHECK(cudaGetLastError());
    } else {
        stage1_binary_ip_lut_flag_docs_v7_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_cagra_labels,
            d_cluster_pos_,
            d_clustered_doc_ids_,
            d_doc_bitmap_,
            num_docs,
            static_cast<int>(nprobe_value),
            ivf->n_clusters);
        CUDA_CHECK(cudaGetLastError());
    }

    bitmap_offset_init_v7_kernel<<<
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
            "v7 compact-doc count " + std::to_string(h_num_touched) +
            " exceeded workspace bound " + std::to_string(max_compact_docs_));
    }

    ensure_compact_doc_capacity(static_cast<size_t>(h_num_touched));
    CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_query_max,
                               0,
                               static_cast<size_t>(h_num_touched) * Q_DOCLEN * sizeof(float),
                               stream));

    bitmap_unique_docs_v7_kernel<<<
        (doc_bitmap_bucket_count_ + threads_per_block - 1) / threads_per_block,
        threads_per_block,
        0,
        stream>>>(
            d_doc_bitmap_,
            doc_bitmap_bucket_count_,
            d_doc_bitmap_offsets_,
            ws_.d_unique_doc_ids);
    CUDA_CHECK(cudaGetLastError());

#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_start, stream));
    }
#endif
    if (stage1_align_mode_ == Stage1AlignMode::MemoryPaddedAlign) {
        stage1_binary_ip_lut_v7_padded_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_lut,
            d_stage1_padded_code_,
            d_stage1_padded_factor_,
            ws_.d_cb1_sumq,
            ws_.d_cagra_labels,
            d_stage1_padded_cluster_meta_,
            d_stage1_padded_doc_ids_,
            ws_.d_doc_query_max,
            d_doc_bitmap_,
            d_doc_bitmap_offsets_,
            h_num_touched,
            num_docs,
            static_cast<int>(nprobe_value),
            ivf->n_clusters);
        CUDA_CHECK(cudaGetLastError());
    } else {
        stage1_binary_ip_lut_v7_kernel<<<grid, threads_per_block, 0, stream>>>(
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
        CUDA_CHECK(cudaGetLastError());
    }
#ifdef CHIMERA_PROFILE
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
#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_sum_scores_start, stream));
    }
#endif
    sum_doc_scores_compact_v7_kernel<<<sparse_blocks, thread_count, 0, stream>>>(
        ws_.d_doc_query_max,
        ws_.d_unique_doc_ids,
        ws_.d_stage1_doc_scores,
        ws_.d_topk_doc_ids,
        h_num_touched);
    CUDA_CHECK(cudaGetLastError());
#ifdef CHIMERA_PROFILE
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
#ifdef CHIMERA_PROFILE
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
#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_d2d_end, stream));
    }
#endif
}


}  // namespace Chimera
