#include "gpu_index_v6.cuh"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>

#include <filesystem>
#include <functional>
#include <system_error>

#include "clustered_format.hpp"
#include "gpu_index_layout.hpp"
#include "doc_format.hpp"
#include "startup_profile.hpp"

#if defined(CHIMERA_CLUSTERED_LAYOUT_DISABLED) && \

namespace Chimera {

    defined(CHIMERA_CLUSTERED_LAYOUT_REQUIRED)
#error "CHIMERA_CLUSTERED_LAYOUT_DISABLED and CHIMERA_CLUSTERED_LAYOUT_REQUIRED are mutually exclusive"
#endif

namespace {

constexpr size_t kCompactDocRowAlignment = 1u << 16;  // 65536 rows
constexpr size_t kCompactDocMinSlackRows = 1u << 15;  // 32768 rows
constexpr size_t kCompactDocBufferAlignment = 128;

size_t sum_top_sorted_counts(std::vector<size_t>& counts, size_t top_n) {
    if (counts.empty() || top_n == 0) {
        return 0;
    }

    std::sort(counts.begin(), counts.end(), std::greater<size_t>());
    top_n = std::min(top_n, counts.size());
    return std::accumulate(counts.begin(), counts.begin() + top_n, size_t{0});
}

std::pair<size_t, size_t> compute_stage2_token_bounds(
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

size_t align_up(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

char* align_up_ptr(char* ptr, size_t alignment) {
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    const auto aligned = align_up(addr, alignment);
    return reinterpret_cast<char*>(aligned);
}

size_t initial_compact_doc_capacity(
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
    estimate = std::min(max_compact_docs, align_up(estimate, kCompactDocRowAlignment));
    return std::max<size_t>(estimate, 1);
}

size_t grow_compact_doc_capacity(size_t current, size_t required, size_t limit) {
    const size_t aligned_required =
        align_up(std::max(required, size_t{1}), kCompactDocRowAlignment);
    const size_t required_slack =
        std::max(required / 16, kCompactDocMinSlackRows);   // ~6.25%
    const size_t current_slack =
        std::max(current / 8, kCompactDocMinSlackRows);      // ~12.5%
    size_t target = aligned_required;
    target = std::max(
        target,
        align_up(required + required_slack, kCompactDocRowAlignment));
    if (current > 0) {
        target = std::max(
            target,
            align_up(current + current_slack, kCompactDocRowAlignment));
    }
    return std::min(limit, target);
}

#ifdef CHIMERA_COMPACT_DOC_BUFFER
size_t compact_doc_arena_bytes(size_t row_capacity, size_t topk_capacity) {
    size_t offset = 0;
    offset = align_up(offset, kCompactDocBufferAlignment);
    offset += row_capacity * sizeof(int);     // d_sorted_doc_ids
    offset = align_up(offset, kCompactDocBufferAlignment);
    offset += row_capacity * sizeof(int);     // d_unique_doc_ids
    offset = align_up(offset, kCompactDocBufferAlignment);
    offset += row_capacity * sizeof(float);   // d_stage1_doc_scores
    offset = align_up(offset, kCompactDocBufferAlignment);
    offset += row_capacity * Q_DOCLEN * sizeof(float);  // d_doc_query_max
    offset = align_up(offset, kCompactDocBufferAlignment);
    offset += topk_capacity * sizeof(float);  // d_topk_scores
    offset = align_up(offset, kCompactDocBufferAlignment);
    offset += topk_capacity * sizeof(int);    // d_topk_doc_ids
    offset = align_up(offset, kCompactDocBufferAlignment);
    offset += topk_capacity * sizeof(int);    // d_topk_indices
    return offset;
}

void check_compact_doc_alignment(const void* ptr, const char* name) {
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    if (addr % kCompactDocBufferAlignment != 0) {
        throw std::runtime_error(
            std::string("Compact doc buffer ") + name +
            " is not 128-byte aligned");
    }
}

void bind_compact_doc_arena(
    chimera_index::Workspace& ws,
    char* base,
    size_t row_capacity,
    size_t topk_capacity) {
    size_t offset = 0;
    offset = align_up(offset, kCompactDocBufferAlignment);
    ws.d_sorted_doc_ids = reinterpret_cast<int*>(base + offset);
    offset += row_capacity * sizeof(int);
    offset = align_up(offset, kCompactDocBufferAlignment);
    ws.d_unique_doc_ids = reinterpret_cast<int*>(base + offset);
    offset += row_capacity * sizeof(int);
    offset = align_up(offset, kCompactDocBufferAlignment);
    ws.d_stage1_doc_scores = reinterpret_cast<float*>(base + offset);
    offset += row_capacity * sizeof(float);
    offset = align_up(offset, kCompactDocBufferAlignment);
    ws.d_doc_query_max = reinterpret_cast<float*>(base + offset);
    offset += row_capacity * Q_DOCLEN * sizeof(float);
    offset = align_up(offset, kCompactDocBufferAlignment);
    ws.d_topk_scores = reinterpret_cast<float*>(base + offset);
    offset += topk_capacity * sizeof(float);
    offset = align_up(offset, kCompactDocBufferAlignment);
    ws.d_topk_doc_ids = reinterpret_cast<int*>(base + offset);
    offset += topk_capacity * sizeof(int);
    offset = align_up(offset, kCompactDocBufferAlignment);
    ws.d_topk_indices = reinterpret_cast<int*>(base + offset);

    check_compact_doc_alignment(base, "base");
    check_compact_doc_alignment(ws.d_sorted_doc_ids, "d_sorted_doc_ids");
    check_compact_doc_alignment(ws.d_unique_doc_ids, "d_unique_doc_ids");
    check_compact_doc_alignment(ws.d_stage1_doc_scores, "d_stage1_doc_scores");
    check_compact_doc_alignment(ws.d_doc_query_max, "d_doc_query_max");
    check_compact_doc_alignment(ws.d_topk_scores, "d_topk_scores");
    check_compact_doc_alignment(ws.d_topk_doc_ids, "d_topk_doc_ids");
    check_compact_doc_alignment(ws.d_topk_indices, "d_topk_indices");
}
#endif

bool env_flag_enabled(const char* name) {
    const char* env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && std::string(env) != "0";
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
        doc_ptrs, 0, static_cast<int>(doc_ptrs.size()) - 2, token_id);
}

std::vector<int> build_doc_block_lut_host(
    const std::vector<int>& doc_ptrs,
    size_t num_docs,
    size_t num_tokens) {
    const size_t num_doc_blocks =
        (num_tokens + kDocPtrLookupBlockSize - 1) >> kDocPtrLookupBlockShift;
    std::vector<int> doc_block_lut(num_doc_blocks + 1, 0);
    size_t doc_id = 0;
    for (size_t block = 0; block < num_doc_blocks; ++block) {
        const size_t token_boundary = block * kDocPtrLookupBlockSize;
        while (doc_id + 1 < doc_ptrs.size() &&
               static_cast<size_t>(doc_ptrs[doc_id + 1]) <= token_boundary) {
            ++doc_id;
        }
        doc_block_lut[block] = static_cast<int>(doc_id);
    }
    doc_block_lut[num_doc_blocks] =
        static_cast<int>(std::max<size_t>(num_docs, 1) - 1);
    return doc_block_lut;
}

}  // namespace

// ======================== CONSTRUCTOR ========================

chimera_index::chimera_index(
    const std::string& filename,
    const std::vector<int>& doc_lens,
    const gpu_search_runtime_options& runtime_options) {
    auto log_ctor_step = [](const std::string& message) {
        std::cout << "[INIT][index_ctor] " << message << std::endl;
    };

    nprobe = runtime_options.nprobe;
    k_rank_cluster = runtime_options.k_rank_cluster;
    k_rank_all_tokens = runtime_options.k_rank_all_tokens;
    itopk_size = runtime_options.itopk_size;
    overlap_chunks = runtime_options.overlap_chunks;

    Chimera::StartupProfile startup("index_ctor");
    const auto resolved_paths = gpu_index_layout::resolve_index_paths(filename);
    startup.mark("resolve_index_paths");

    log_ctor_step(
        "Loading quantized payloads from " +
        resolved_paths.quantized_data_path + " and " +
        resolved_paths.doc_4bit_path + " ...");
    std::ifstream inf(resolved_paths.quantized_data_path, std::ios::binary);
    const auto header =
        doc_format::read_header(inf, resolved_paths.quantized_data_path);
    n = header.n;
    d = header.d;
    n_clusters = header.n_clusters;
    ex_bits = header.ex_bits;

#ifdef CHIMERA_V6_GPU_INDICES_U32
    if (n > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error(
            "gpu_search_v6 built with CHIMERA_V6_GPU_INDICES_U32=1, which "
            "limits the total token count to 4,294,967,295, but index "
            "embedding count exceeds that capacity: n=" +
            std::to_string(n));
    }
#endif

    if (header.padded_dim != PADDED_DIM) {
        inf.close();
        throw std::runtime_error(
            "Index file padded_dim=" + std::to_string(header.padded_dim) +
            " does not match compiled PADDED_DIM=" + std::to_string(PADDED_DIM) +
            ". Please recompile with matching PADDED_DIM in gpu_config.cuh"
        );
    }

    rotator_ = doc_format::load_rotator(inf, header, filename);
    startup.mark("read_header_and_rotator");

    const size_t one_bit_bytes = n * (PADDED_DIM / 8);
    one_bit_code_.resize(one_bit_bytes);
    full_code_.resize(n * PADDED_DIM * (1 + ex_bits) / 8);
    one_bit_factor_.resize(n);
    ex_factor_.resize(n);

    inf.read(one_bit_code_.data(), one_bit_code_.size());
    inf.read(reinterpret_cast<char*>(one_bit_factor_.data()), n * sizeof(float));
    inf.read(reinterpret_cast<char*>(ex_factor_.data()), n * sizeof(float));
    if (!inf) {
        throw std::runtime_error(
            "Failed to read doc_1bit payload from " + resolved_paths.quantized_data_path);
    }
    inf.close();

    std::ifstream doc4(resolved_paths.doc_4bit_path, std::ios::binary);
    const auto doc4_header =
        doc_format::read_header(doc4, resolved_paths.doc_4bit_path);
    doc_format::validate_matching_header(
        header,
        doc4_header,
        resolved_paths.quantized_data_path,
        resolved_paths.doc_4bit_path);
    auto* doc4_rotator =
        doc_format::load_rotator(doc4, doc4_header, resolved_paths.doc_4bit_path);
    delete doc4_rotator;
    doc4.read(full_code_.data(), full_code_.size());
    if (!doc4) {
        throw std::runtime_error(
            "Failed to read doc_4bit payload from " + resolved_paths.doc_4bit_path);
    }
    startup.mark("read_quantized_payload_and_factors");

    ip_func_ = select_excode_ipfunc(1 + ex_bits);
    unpack_func_ = select_excode_unpackfunc(1 + ex_bits);

    log_ctor_step(
        "Loading IVF postings and centroid graph from " +
        resolved_paths.ivf_path + " and " +
        resolved_paths.centroids_path + " ...");
    ivf = new IVF_PG(n_clusters, d, PGType::CAGRA);
    ivf->load(resolved_paths.ivf_path, resolved_paths.centroids_path);
    max_cluster_size = ivf->max_cluster_size();
    std::cout << "max cluster size: " << max_cluster_size << std::endl;
    startup.mark("load_ivf");

    use_docptr_remap_ = env_flag_enabled("CHIMERA_V6_USE_DOCPTRS");
    if (use_docptr_remap_) {
        std::cout
            << "[chimera][v6] CHIMERA_V6_USE_DOCPTRS is set. "
               "Skipping d_doc_ids_ and using doc_ptrs block LUT remapping "
               "in non-clustered stage 1."
            << std::endl;
    }

    log_ctor_step(
        "Building document-to-token map for " +
        std::to_string(doc_lens.size()) + " docs and " +
        std::to_string(n) + " embeddings ...");
    set_doc_mapping(doc_lens);
    startup.mark("set_doc_mapping");
    log_ctor_step(
        "Computing top-" + std::to_string(nprobe) +
        " per-query workspace bounds from cluster token/doc counts ...");
    compute_workspace_probe_bounds();
    startup.mark("compute_workspace_probe_bounds");
    log_ctor_step("Uploading persistent stage-1 data to GPU ...");
    const size_t code_bytes = n * PADDED_DIM / 8;
    CUDA_CHECK(cudaMalloc(&d_one_bit_code_, code_bytes));
    CUDA_CHECK(cudaMalloc(&d_one_bit_factor_, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_doc_ptrs_, (num_docs + 1) * sizeof(int)));
    if (!use_docptr_remap_) {
        CUDA_CHECK(cudaMalloc(&d_doc_ids_, n * sizeof(int)));
    }

    std::vector<int> doc_block_lut;
    if (use_docptr_remap_) {
        doc_block_lut = build_doc_block_lut_host(doc_ptrs_, num_docs, n);
        CUDA_CHECK(cudaMalloc(
            &d_doc_block_lut_,
            doc_block_lut.size() * sizeof(int)));
    }

#ifdef CHIMERA_V6_GPU_INDICES_U32
    std::vector<gpu_cluster_pos_t> cluster_pos_gpu(ivf->n_clusters + 1, 0);
    for (size_t i = 0; i < ivf->cluster_pos.size(); ++i) {
        cluster_pos_gpu[i] =
            static_cast<gpu_cluster_pos_t>(ivf->cluster_pos[i]);
    }
    CUDA_CHECK(cudaMalloc(
        &d_cluster_pos_,
        (ivf->n_clusters + 1) * sizeof(gpu_cluster_pos_t)));
    CUDA_CHECK(cudaMemcpy(
        d_cluster_pos_,
        cluster_pos_gpu.data(),
        (ivf->n_clusters + 1) * sizeof(gpu_cluster_pos_t),
        cudaMemcpyHostToDevice));
#else
    CUDA_CHECK(cudaMalloc(&d_cluster_pos_, (ivf->n_clusters + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(
        d_cluster_pos_,
        ivf->cluster_pos.data(),
        (ivf->n_clusters + 1) * sizeof(size_t),
        cudaMemcpyHostToDevice));
#endif

#ifndef CHIMERA_USE_LUT
    {
        size_t inv_list_size = ivf->inv_list.size();
        CUDA_CHECK(cudaMalloc(&d_inv_list_, inv_list_size * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            d_inv_list_,
            ivf->inv_list.data(),
            inv_list_size * sizeof(int),
            cudaMemcpyHostToDevice));
    }
#else
    d_inv_list_ = nullptr;
#endif

    CUDA_CHECK(cudaMemcpy(d_one_bit_code_, one_bit_code_.data(), code_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_one_bit_factor_, one_bit_factor_.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_doc_ptrs_, doc_ptrs_.data(), (num_docs + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (!use_docptr_remap_) {
        CUDA_CHECK(cudaMemcpy(
            d_doc_ids_,
            doc_ids_.data(),
            n * sizeof(int),
            cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(
            d_doc_block_lut_,
            doc_block_lut.data(),
            doc_block_lut.size() * sizeof(int),
            cudaMemcpyHostToDevice));
    }
    startup.mark("upload_persistent_data");

    log_ctor_step(
        "Allocating GPU workspace for max runtime config "
        "(nprobe=" + std::to_string(nprobe) +
        ", k_rank_cluster=" + std::to_string(k_rank_cluster) +
        ", k_rank_all_tokens=" + std::to_string(k_rank_all_tokens) +
        ", itopk_size=" + std::to_string(itopk_size) +
        ", overlap_chunks=" + std::to_string(overlap_chunks) + ") ...");
    allocate_workspace();
    startup.mark("allocate_workspace");

    {
        const size_t inv_n = ivf->inv_list.size();
        const size_t code_per_vec = PADDED_DIM / 8;
#ifdef CHIMERA_CLUSTERED_LAYOUT_DISABLED
#ifdef CHIMERA_USE_LUT
        if (d_inv_list_ == nullptr) {
            const size_t inv_list_size = ivf->inv_list.size();
            CUDA_CHECK(cudaMalloc(&d_inv_list_, inv_list_size * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(
                d_inv_list_,
                ivf->inv_list.data(),
                inv_list_size * sizeof(int),
                cudaMemcpyHostToDevice));
        }
#endif
        use_clustered_ = false;
        d_clustered_code_ = nullptr;
        d_clustered_factor_ = nullptr;
        d_clustered_doc_ids_ = nullptr;
        std::cout << "[chimera] Clustered layout disabled for this build. "
                     "Using original-order layout with inv_list indirection."
                  << std::endl;
        startup.mark("disable_clustered_layout");
#else
        const size_t clustered_bytes =
            inv_n * (code_per_vec + sizeof(float) + sizeof(int));

        auto ensure_inv_list_on_gpu = [&]() {
#ifdef CHIMERA_USE_LUT
            if (d_inv_list_ == nullptr) {
                const size_t inv_list_size = ivf->inv_list.size();
                CUDA_CHECK(cudaMalloc(&d_inv_list_, inv_list_size * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(
                    d_inv_list_,
                    ivf->inv_list.data(),
                    inv_list_size * sizeof(int),
                    cudaMemcpyHostToDevice));
            }
#endif
        };

        auto disable_clustered_layout = [&](const std::string& reason, const char* startup_label) {
            use_clustered_ = false;
            d_clustered_code_ = nullptr;
            d_clustered_factor_ = nullptr;
            d_clustered_doc_ids_ = nullptr;
            ensure_inv_list_on_gpu();
            std::cout << reason << std::endl;
            startup.mark(startup_label);
        };

        auto load_persisted_clustered_layout = [&]() {
            if (resolved_paths.gpu_index_path.empty() ||
                !std::filesystem::exists(resolved_paths.gpu_index_path)) {
                throw std::runtime_error(
                    "Required clustered sidecar is missing: " +
                    resolved_paths.gpu_index_path);
            }

            log_ctor_step(
                "Loading cluster-ordered stage-1 layout from " +
                resolved_paths.gpu_index_path + " ...");
            std::ifstream clustered_header_file(
                resolved_paths.gpu_index_path, std::ios::binary);
            const auto clustered_header =
                clustered_format::read_header(
                    clustered_header_file, resolved_paths.gpu_index_path);
            if (clustered_header.n_entries != inv_n ||
                clustered_header.code_bytes_per_vector != code_per_vec) {
                throw std::runtime_error(
                    "GPU index metadata mismatch for " +
                    resolved_paths.gpu_index_path);
            }
            if (clustered_format::has_embedded_rotator(clustered_header)) {
                if (clustered_header.source_dim != header.d ||
                    clustered_header.padded_dim != header.padded_dim ||
                    clustered_header.rotator_type != header.rotator_type) {
                    throw std::runtime_error(
                        "cluster_1bit rotator metadata mismatch for " +
                        resolved_paths.gpu_index_path);
                }
                auto* clustered_rotator = clustered_format::load_rotator(
                    clustered_header_file,
                    clustered_header,
                    resolved_paths.gpu_index_path);
                delete clustered_rotator;
            }
            clustered_header_file.close();

            struct stat clustered_stat {};
            const int clustered_fd = open(resolved_paths.gpu_index_path.c_str(), O_RDONLY);
            if (clustered_fd < 0) {
                throw std::runtime_error(
                    "Failed to open GPU index: " + resolved_paths.gpu_index_path);
            }
            if (fstat(clustered_fd, &clustered_stat) != 0) {
                const auto err = errno;
                close(clustered_fd);
                throw std::runtime_error(
                    "Failed to stat GPU index " + resolved_paths.gpu_index_path +
                    ": " + std::system_category().message(err));
            }

            const size_t mapping_size = static_cast<size_t>(clustered_stat.st_size);
            const size_t expected_size =
                clustered_format::prefix_bytes(clustered_header) +
                inv_n * code_per_vec +
                inv_n * sizeof(float) +
                inv_n * sizeof(int) +
                inv_n * sizeof(uint32_t);
            if (mapping_size != expected_size) {
                close(clustered_fd);
                throw std::runtime_error(
                    "Unexpected GPU index size for " +
                    resolved_paths.gpu_index_path);
            }

            void* clustered_mapping =
                mmap(nullptr, mapping_size, PROT_READ, MAP_PRIVATE, clustered_fd, 0);
            if (clustered_mapping == MAP_FAILED) {
                const auto err = errno;
                close(clustered_fd);
                throw std::runtime_error(
                    "Failed to mmap GPU index " + resolved_paths.gpu_index_path +
                    ": " + std::system_category().message(err));
            }

            const auto* clustered_base = static_cast<const char*>(clustered_mapping);
            const size_t clustered_prefix_bytes =
                clustered_format::prefix_bytes(clustered_header);
            const auto* clustered_code =
                clustered_base + clustered_prefix_bytes;
            const auto* clustered_factor = reinterpret_cast<const float*>(
                clustered_code + inv_n * code_per_vec);
            const auto* clustered_doc_ids = reinterpret_cast<const int*>(
                reinterpret_cast<const char*>(clustered_factor) +
                inv_n * sizeof(float));

            CUDA_CHECK(cudaMalloc(&d_clustered_code_, inv_n * code_per_vec));
            CUDA_CHECK(cudaMalloc(&d_clustered_factor_, inv_n * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_clustered_doc_ids_, inv_n * sizeof(int)));

            CUDA_CHECK(cudaMemcpy(
                d_clustered_code_,
                clustered_code,
                inv_n * code_per_vec,
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_clustered_factor_,
                clustered_factor,
                inv_n * sizeof(float),
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_clustered_doc_ids_,
                clustered_doc_ids,
                inv_n * sizeof(int),
                cudaMemcpyHostToDevice));

            munmap(clustered_mapping, mapping_size);
            close(clustered_fd);
            use_clustered_ = true;
            startup.note("clustered_layout_source", "persisted");
            std::cout << "[chimera] Using cluster-ordered layout ("
                      << (clustered_bytes / (1024.0 * 1024.0)) << " MB)"
                      << std::endl;
            startup.mark("load_clustered_layout");
        };

#ifdef CHIMERA_CLUSTERED_LAYOUT_REQUIRED
        size_t free_mem = 0;
        size_t total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        if (free_mem <= clustered_bytes) {
            throw std::runtime_error(
                "Insufficient GPU memory for required cluster_1bit.bin layout: need " +
                std::to_string(clustered_bytes / (1024 * 1024)) +
                " MB, have " +
                std::to_string(free_mem / (1024 * 1024)) +
                " MB free");
        }
        load_persisted_clustered_layout();
#else
        size_t free_mem = 0;
        size_t total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

        const bool force_nonclustered =
            [] {
                const char* env = std::getenv("CHIMERA_FORCE_NONCLUSTERED");
                return env != nullptr && env[0] != '\0' && std::string(env) != "0";
            }();

        if (!force_nonclustered && free_mem > clustered_bytes) {
            load_persisted_clustered_layout();
        } else {
            if (force_nonclustered) {
                disable_clustered_layout(
                    "[chimera] CHIMERA_FORCE_NONCLUSTERED is set. "
                    "Using original-order layout with inv_list indirection.",
                    "skip_clustered_layout");
            } else {
                disable_clustered_layout(
                    "[chimera] Insufficient GPU memory for cluster-ordered layout (" +
                    std::to_string(clustered_bytes / (1024 * 1024)) +
                    " MB needed, " +
                    std::to_string(free_mem / (1024 * 1024)) +
                    " MB free). Falling back to inv_list indirection.",
                "skip_clustered_layout");
            }
        }
#endif
#endif
    }
}

// ======================== set_doc_mapping ========================

void chimera_index::set_doc_mapping(const std::vector<int>& doc_lens) {
    num_docs = doc_lens.size();
    doc_ptrs_.resize(num_docs + 1, 0);
    for (size_t i = 0; i < num_docs; ++i) {
        max_doc_len = std::max(max_doc_len, doc_lens[i]);
        doc_ptrs_[i + 1] = doc_ptrs_[i] + doc_lens[i];
    }
    if (!use_docptr_remap_) {
        doc_ids_.resize(n);
        for (size_t i = 0; i < num_docs; ++i) {
            for (size_t j = 0; j < doc_lens[i]; ++j) {
                doc_ids_[doc_ptrs_[i] + j] = i;
            }
        }
    }
    if (static_cast<size_t>(doc_ptrs_.back()) != n) {
        throw std::runtime_error(
            "doclens token count " + std::to_string(doc_ptrs_.back()) +
            " does not match index embedding count " + std::to_string(n));
    }
}

void chimera_index::compute_workspace_probe_bounds() {
    const size_t probe_cluster_bound =
        std::min(static_cast<size_t>(std::max(nprobe, 0)), n_clusters);
    workspace_probe_cluster_bound_ = probe_cluster_bound;
    if (probe_cluster_bound == 0) {
        workspace_probe_token_bound_ = 0;
#ifdef CHIMERA_COMPACT_DOC_BUFFER
        workspace_probe_unique_doc_bound_ = 0;
#endif
        return;
    }

    std::vector<size_t> cluster_token_counts(n_clusters, 0);
#ifdef CHIMERA_COMPACT_DOC_BUFFER
    std::vector<size_t> cluster_unique_doc_counts(n_clusters, 0);
#endif

#pragma omp parallel for schedule(dynamic, 256)
    for (int cluster_idx = 0; cluster_idx < static_cast<int>(n_clusters); ++cluster_idx) {
        const size_t cluster_id = static_cast<size_t>(cluster_idx);
        const size_t start = ivf->cluster_pos[cluster_id];
        const size_t end = ivf->cluster_pos[cluster_id + 1];
        const size_t token_count = end - start;
        cluster_token_counts[cluster_id] = token_count;

#ifdef CHIMERA_COMPACT_DOC_BUFFER
        if (token_count == 0) {
            cluster_unique_doc_counts[cluster_id] = 0;
            continue;
        }

        const auto first_token_id = static_cast<uint32_t>(ivf->inv_list[start]);
        int prev_doc_id = use_docptr_remap_
            ? doc_id_from_doc_ptrs_host(doc_ptrs_, first_token_id)
            : doc_ids_[first_token_id];
        size_t unique_doc_count = 1;

        for (size_t pos = start + 1; pos < end; ++pos) {
            const uint32_t token_id = static_cast<uint32_t>(ivf->inv_list[pos]);
            const int current_doc_id = use_docptr_remap_
                ? doc_id_from_doc_ptrs_host_binary_range(
                    doc_ptrs_, prev_doc_id, static_cast<int>(num_docs) - 1, token_id)
                : doc_ids_[token_id];
            if (current_doc_id != prev_doc_id) {
                ++unique_doc_count;
                prev_doc_id = current_doc_id;
            }
        }

        cluster_unique_doc_counts[cluster_id] = unique_doc_count;
#endif
    }

    workspace_probe_token_bound_ =
        sum_top_sorted_counts(cluster_token_counts, probe_cluster_bound);
#ifdef CHIMERA_COMPACT_DOC_BUFFER
    workspace_probe_unique_doc_bound_ =
        sum_top_sorted_counts(cluster_unique_doc_counts, probe_cluster_bound);
#endif
}

// ======================== allocate_workspace ========================

void chimera_index::allocate_workspace() {
    ws_.max_q_doclen = Q_DOCLEN;
    ws_.max_stage1_pairs = workspace_probe_token_bound_ * Q_DOCLEN;
    ws_.max_stage2_candidates = k_rank_cluster;
    ws_.max_stage2_k = k_rank_all_tokens;
    std::tie(ws_.max_stage2_tokens, ws_.max_stage2_k_tokens) =
        compute_stage2_token_bounds(
            doc_ptrs_, ws_.max_stage2_candidates, ws_.max_stage2_k);
    ws_.estimated_num_docs = (size_t)num_docs;
    ws_.max_stage1_touched_docs =
        std::min(ws_.estimated_num_docs, ws_.max_stage1_pairs);

#ifdef CHIMERA_COMPACT_DOC_BUFFER
    // The compact-doc buffer tracks unique documents, not tokens.
    // A safe per-query-token upper bound is the sum of the largest nprobe
    // per-cluster unique-doc counts. Across Q_DOCLEN query tokens, multiply
    // that bound by Q_DOCLEN and clamp to num_docs.
    ws_.max_compact_docs = std::min(
        (size_t)num_docs,
        workspace_probe_unique_doc_bound_ * Q_DOCLEN);
    ws_.doc_bitmap_bucket_count = doc_bitmap_num_buckets(num_docs);
    ws_.doc_bitmap_offset_count =
        doc_bitmap_num_offsets(ws_.doc_bitmap_bucket_count);
    ws_.compact_doc_capacity = initial_compact_doc_capacity(
        n,
        n_clusters,
        workspace_probe_cluster_bound_,
        ws_.max_compact_docs);
    const size_t doc_buf_rows = ws_.compact_doc_capacity;

    std::cout << "max cluster size: " << max_cluster_size << std::endl;
    std::cout << "workspace_probe_cluster_bound: "
              << workspace_probe_cluster_bound_ << std::endl;
    std::cout << "workspace_probe_token_bound_per_query: "
              << workspace_probe_token_bound_ << std::endl;
    std::cout << "workspace_probe_unique_doc_bound_per_query: "
              << workspace_probe_unique_doc_bound_ << std::endl;
    std::cout << "max_compact_docs: " << ws_.max_compact_docs
              << "  (num_docs=" << num_docs << ")" << std::endl;
    std::cout << "initial_compact_doc_capacity: "
              << ws_.compact_doc_capacity
              << "  (4 * avg_cluster_tokens * max_nprobe, then grows to observed touched docs)"
              << std::endl;
    std::cout << "workspace_stage2_candidate_token_bound: "
              << ws_.max_stage2_tokens << std::endl;
    std::cout << "workspace_stage2_topk_token_bound: "
              << ws_.max_stage2_k_tokens << std::endl;
    std::cout << "bitmap buckets: " << ws_.doc_bitmap_bucket_count
              << " bitmap offsets: " << ws_.doc_bitmap_offset_count
              << std::endl;
    const size_t doc_query_rows = doc_buf_rows;
#else
    const size_t doc_buf_rows = std::max<size_t>(ws_.max_stage1_touched_docs, 1);
    const size_t doc_query_rows = std::max<size_t>(ws_.estimated_num_docs, 1);
    std::cout << "max cluster size: " << max_cluster_size << std::endl;
    std::cout << "workspace_probe_cluster_bound: "
              << workspace_probe_cluster_bound_ << std::endl;
    std::cout << "workspace_probe_token_bound_per_query: "
              << workspace_probe_token_bound_ << std::endl;
    std::cout << "workspace_stage1_doc_bound: "
              << ws_.max_stage1_touched_docs << std::endl;
    std::cout << "workspace_stage2_candidate_token_bound: "
              << ws_.max_stage2_tokens << std::endl;
    std::cout << "workspace_stage2_topk_token_bound: "
              << ws_.max_stage2_k_tokens << std::endl;
#endif

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

    size_t estimated_size = Q_DOCLEN * PADDED_DIM * sizeof(float) +  // d_queries
                            Q_DOCLEN * sizeof(float) +  // d_cb1_sumq
                            stage1_emb_ids_bytes +  // d_emb_ids
                            stage1_pair_offsets_bytes +  // d_pair_offsets
                            stage1_emb_dists_bytes +  // d_emb_dists
                            phase_a_doc_lengths_bytes +  // d_pair_doc_ids
                            doc_buf_rows * sizeof(int) +  // d_sorted_doc_ids
                            doc_buf_rows * sizeof(int) +  // d_unique_doc_ids
                            doc_buf_rows * sizeof(float) +  // d_stage1_doc_scores
                            doc_query_rows * Q_DOCLEN * sizeof(float) +  // d_doc_query_max
                            sizeof(int) +  // d_num_unique_docs
#ifdef CHIMERA_COMPACT_DOC_BUFFER
                            ws_.doc_bitmap_bucket_count * sizeof(doc_bitmap_bucket_t) +
                            ws_.doc_bitmap_offset_count * sizeof(doc_bitmap_offset_t);
#else
                            ws_.estimated_num_docs * sizeof(int);  // d_doc_touched
#endif
    std::cout << "Initial GPU memory required for workspace: "
              << (estimated_size / (1024.0 * 1024.0)) << " MB" << std::endl;

    CUDA_CHECK(cudaMalloc(&ws_.d_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_cb1_sumq, Q_DOCLEN * sizeof(float)));

#ifndef CHIMERA_USE_LUT
    CUDA_CHECK(cudaMalloc(&ws_.d_emb_ids, ws_.max_stage1_pairs * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_offsets, (Q_DOCLEN + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_emb_dists, ws_.max_stage1_pairs * sizeof(float)));
#endif
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_doc_ids, ws_.max_stage2_candidates * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&ws_.d_num_unique_docs, sizeof(int)));
#ifdef CHIMERA_COMPACT_DOC_BUFFER
    CUDA_CHECK(cudaMalloc(&ws_.d_doc_bitmap,
                          ws_.doc_bitmap_bucket_count * sizeof(doc_bitmap_bucket_t)));
    CUDA_CHECK(cudaMalloc(&ws_.d_doc_bitmap_offsets,
                          ws_.doc_bitmap_offset_count * sizeof(doc_bitmap_offset_t)));
#else
    CUDA_CHECK(cudaMalloc(&ws_.d_doc_touched, ws_.estimated_num_docs * sizeof(int)));
#endif

    size_t free_mem, total_mem;
    // CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // std::cout << "GPU memory usage after index upload: "
    //           << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB / "
    //           << (total_mem / (1024.0 * 1024.0)) << " MB\n";

    ws_.d_cub_temp_storage = nullptr;
    ws_.cub_temp_storage_bytes = 0;

    size_t temp_bytes = 0;

    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, temp_bytes,
        ws_.d_stage1_doc_scores, ws_.d_topk_scores,
        ws_.d_topk_indices, ws_.d_sorted_doc_ids,
        (int)doc_buf_rows, 0, 32
    );
    ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);

#ifndef CHIMERA_OVERLAP_STAGE23
    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, temp_bytes,
        ws_.d_doc_scores, ws_.d_topk_scores,
        ws_.d_selected_indices, ws_.d_topk_indices,
        (int)ws_.max_stage2_candidates, 0, 32
    );
    ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);
#endif

#ifndef CHIMERA_USE_LUT
    temp_bytes = 0;
    cub::DeviceScan::InclusiveScan(
        nullptr, temp_bytes,
        ws_.d_pair_offsets + 1, ws_.d_pair_offsets + 1,
        thrust::plus<size_t>(), (int)Q_DOCLEN
    );
    ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);
#endif

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

#ifdef CHIMERA_COMPACT_DOC_BUFFER
    temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_bytes,
        ws_.d_doc_bitmap_offsets,
        ws_.d_doc_bitmap_offsets,
        ws_.doc_bitmap_offset_count);
    ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);
#endif

    CUDA_CHECK(cudaMalloc(&ws_.d_cub_temp_storage, ws_.cub_temp_storage_bytes));

    // CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // std::cout << "GPU memory usage after index upload: "
    //           << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB / "
    //           << (total_mem / (1024.0 * 1024.0)) << " MB\n";

#ifdef CHIMERA_COMPACT_DOC_BUFFER
    ws_.d_compact_doc_buffer_raw = nullptr;
    ws_.d_compact_doc_buffer = nullptr;
    ws_.compact_doc_buffer_bytes = 0;
    ws_.compact_doc_capacity = 0;
    ws_.topk_buf_capacity = 0;
    ensure_compact_doc_capacity(doc_buf_rows);
#else
    CUDA_CHECK(cudaMalloc(&ws_.d_sorted_doc_ids, doc_buf_rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_unique_doc_ids, doc_buf_rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_stage1_doc_scores, doc_buf_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_doc_query_max, doc_query_rows * Q_DOCLEN * sizeof(float)));
    size_t topk_buf_size = std::max(doc_buf_rows, ws_.max_stage2_candidates);
    ws_.topk_buf_capacity = topk_buf_size;
    CUDA_CHECK(cudaMalloc(&ws_.d_topk_scores, topk_buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_topk_doc_ids, topk_buf_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_topk_indices, topk_buf_size * sizeof(int)));
#endif

#ifndef CHIMERA_OVERLAP_STAGE23
    CUDA_CHECK(cudaMalloc(&ws_.d_candidate_offsets, (ws_.max_stage2_candidates + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&ws_.d_token_dists, ws_.max_stage2_tokens * Q_DOCLEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_doc_scores, ws_.max_stage2_candidates * sizeof(float)));
#endif

    CUDA_CHECK(cudaMalloc(&ws_.d_selected_indices, ws_.max_stage2_k * sizeof(int)));

#ifdef CHIMERA_USE_LUT
    CUDA_CHECK(cudaMalloc(&ws_.d_lut, LUT_TOTAL_FLOATS * sizeof(float)));
    size_t stage2_lut_smem = STAGE2_LUT_SMEM_FLOATS * sizeof(float) + STAGE2_LUT_TILE_Q * sizeof(float);
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
    CUDA_CHECK(cudaMalloc(&ws_.d_pst_token_ids,
                          ws_.max_stage2_tokens * sizeof(size_t)));

    CUDA_CHECK(cudaMalloc(&ws_.d_token_dists,
                          ws_.max_stage2_tokens * (size_t)Q_DOCLEN * sizeof(float)));
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
    CUDA_CHECK(cudaEventCreateWithFlags(&ws_.pst_extract_done, cudaEventDisableTiming));
    for (int i = 0; i < Workspace::PST_NUM_D2H_CHUNKS; i++)
        CUDA_CHECK(cudaEventCreateWithFlags(&ws_.pst_d2h_chunk_done[i], cudaEventDisableTiming));
#endif

    ws_.max_embs_per_query_bound = static_cast<int>(workspace_probe_token_bound_);
    CUDA_CHECK(cudaMemset(ws_.d_doc_query_max, 0, doc_buf_rows * Q_DOCLEN * sizeof(float)));
#ifdef CHIMERA_COMPACT_DOC_BUFFER
    CUDA_CHECK(cudaMemset(ws_.d_doc_bitmap, 0,
                          ws_.doc_bitmap_bucket_count * sizeof(doc_bitmap_bucket_t)));
    CUDA_CHECK(cudaMemset(ws_.d_doc_bitmap_offsets, 0,
                          ws_.doc_bitmap_offset_count * sizeof(doc_bitmap_offset_t)));
#else
    CUDA_CHECK(cudaMemset(ws_.d_doc_touched, 0, ws_.estimated_num_docs * sizeof(int)));
#endif
    CUDA_CHECK(cudaMemset(ws_.d_num_unique_docs, 0, sizeof(int)));

    int least_prio, greatest_prio;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio));
    CUDA_CHECK(cudaStreamCreateWithPriority(&ws_.stream_compute, cudaStreamDefault, least_prio));
    CUDA_CHECK(cudaStreamCreate(&ws_.stream_h2d));
    CUDA_CHECK(cudaStreamCreate(&ws_.stream_d2h));
    CUDA_CHECK(cudaStreamCreateWithPriority(&ws_.stream_extract, cudaStreamDefault, greatest_prio));
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
    CUDA_CHECK(cudaEventCreate(&ws_.s2_extract_start));
    CUDA_CHECK(cudaEventCreate(&ws_.s2_extract_end));

    CUDA_CHECK(cudaEventCreate(&ws_.s23_pst_kernel_start));
    CUDA_CHECK(cudaEventCreate(&ws_.s23_pst_kernel_end));

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
    ws_.seen_doc_map.resize(num_docs, false);
}

#ifdef CHIMERA_COMPACT_DOC_BUFFER
void chimera_index::ensure_stage1_sort_capacity(size_t required_rows) {
    required_rows = std::max<size_t>(required_rows, 1);

    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, temp_bytes,
        ws_.d_stage1_doc_scores, ws_.d_topk_scores,
        ws_.d_topk_indices, ws_.d_sorted_doc_ids,
        static_cast<int>(required_rows), 0, 32);
    if (temp_bytes <= ws_.cub_temp_storage_bytes) {
        return;
    }

    CUDA_CHECK(cudaFree(ws_.d_cub_temp_storage));
    ws_.d_cub_temp_storage = nullptr;
    ws_.cub_temp_storage_bytes = temp_bytes;
    CUDA_CHECK(cudaMalloc(&ws_.d_cub_temp_storage, ws_.cub_temp_storage_bytes));
}

void chimera_index::ensure_compact_doc_capacity(size_t required_rows) {
    required_rows = std::max<size_t>(required_rows, 1);
    if (required_rows <= ws_.compact_doc_capacity) {
        return;
    }

    const size_t grown_rows = grow_compact_doc_capacity(
        ws_.compact_doc_capacity,
        required_rows,
        static_cast<size_t>(num_docs));
    const size_t required_topk_capacity =
        std::max(grown_rows, ws_.max_stage2_candidates);
    const size_t required_buffer_bytes =
        compact_doc_arena_bytes(grown_rows, required_topk_capacity);

    std::cout << "[workspace] Growing compact doc capacity from "
              << ws_.compact_doc_capacity << " to "
              << grown_rows << " rows"
              << " (required=" << required_rows << ")" << std::endl;

    CUDA_CHECK(cudaFree(ws_.d_compact_doc_buffer_raw));
    ws_.d_compact_doc_buffer_raw = nullptr;
    ws_.d_compact_doc_buffer = nullptr;
    ws_.compact_doc_buffer_bytes = required_buffer_bytes;
    CUDA_CHECK(cudaMalloc(
        &ws_.d_compact_doc_buffer_raw,
        ws_.compact_doc_buffer_bytes + kCompactDocBufferAlignment - 1));
    ws_.d_compact_doc_buffer =
        align_up_ptr(ws_.d_compact_doc_buffer_raw, kCompactDocBufferAlignment);
    bind_compact_doc_arena(
        ws_, ws_.d_compact_doc_buffer, grown_rows, required_topk_capacity);

    ws_.compact_doc_capacity = grown_rows;
    ws_.topk_buf_capacity = required_topk_capacity;
    ensure_stage1_sort_capacity(grown_rows);
}
#endif

// ======================== doc_len ========================

size_t chimera_index::doc_len(size_t doc_id) const {
    return doc_ptrs_[doc_id + 1] - doc_ptrs_[doc_id];
}

// ======================== SEARCH PIPELINE ========================

std::vector<size_t> chimera_index::search(const float* queries, size_t k) {
    return search_impl<false>(queries, k);
}

std::vector<size_t> chimera_index::search_profiled(const float* queries, size_t k) {
    return search_impl<true>(queries, k, nullptr, true);
}

std::vector<size_t> chimera_index::search_profiled(
    const float* queries,
    size_t k,
    gpu_search_profile_v6* profile,
    bool print_profile) {
    return search_impl<true>(queries, k, profile, print_profile);
}

template <bool kProfile>
std::vector<size_t> chimera_index::search_impl(
    const float* queries,
    size_t k,
    gpu_search_profile_v6* profile,
    bool print_profile) {
    auto search_start = std::chrono::high_resolution_clock::now();
    gpu_search_profile_v6 local_profile;
    gpu_search_profile_v6* profile_ptr = profile;
    if constexpr (kProfile) {
        if (profile_ptr == nullptr && print_profile) {
            profile_ptr = &local_profile;
        }
    }
#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        if (profile_ptr != nullptr) {
            *profile_ptr = gpu_search_profile_v6{};
        }
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
        XFER_RECORD_END(ws_.stream_h2d, Q_DOCLEN * PADDED_DIM * sizeof(float) + Q_DOCLEN * sizeof(float), true);
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
    rank_cluster_dists_gpu_impl<kProfile>(query_objs.data(), nprobe, k_rank_cluster,
                                          actual_k_stage1, ws_.stream_compute);

#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.event_stage1_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventRecord(ws_.event_stage2_start, ws_.stream_compute));
    }
#endif

    std::vector<size_t> result;

#ifdef CHIMERA_OVERLAP_STAGE23
    rank_stage23_persistent_impl<kProfile>(
        actual_k_stage1,
        k,
        k_rank_all_tokens,
        query_objs.data(),
        result,
        profile_ptr,
        false);

#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
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
        float s1_sum = s1_cagra + s1_expansion + s1_binary_ip + s1_atomic_agg + s1_sum_scores + s1_topk_sort + s1_d2d;

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

        if (profile_ptr != nullptr) {
            profile_ptr->persistent_stage23 = true;
            profile_ptr->total_search_time_ms = total_time;
            profile_ptr->stage1_time_ms = stage1_time;
            profile_ptr->s1_cagra_ms = s1_cagra;
            profile_ptr->s1_expansion_ms = s1_expansion;
            profile_ptr->s1_binary_ip_ms = s1_binary_ip;
            profile_ptr->s1_atomic_agg_ms = s1_atomic_agg;
            profile_ptr->s1_sum_scores_ms = s1_sum_scores;
            profile_ptr->s1_topk_sort_ms = s1_topk_sort;
            profile_ptr->s1_d2d_ms = s1_d2d;
            profile_ptr->s1_memset_ms = s1_memset;
            profile_ptr->s1_sum_accounted_ms = s1_sum;
            profile_ptr->phase_b_binary_ip_total_ms = ws_.s23_pst_bip_ms;
            profile_ptr->phase_b_doc_score_total_ms = ws_.s23_pst_docscore_ms;
            profile_ptr->phase_b_total_kernel_ms = ws_.s23_pst_kernel_ms;
            profile_ptr->total_h2d_ms = total_h2d_ms;
            profile_ptr->total_d2h_ms = total_d2h_ms;
            profile_ptr->total_transfer_ms = total_h2d_ms + total_d2h_ms;
            profile_ptr->total_h2d_bytes = static_cast<double>(total_h2d_bytes);
            profile_ptr->total_d2h_bytes = static_cast<double>(total_d2h_bytes);
            profile_ptr->total_transfer_bytes =
                static_cast<double>(total_h2d_bytes + total_d2h_bytes);
        }
    }
#endif

#else  // !CHIMERA_OVERLAP_STAGE23
    std::vector<size_t> stage2_doc_ids;
    std::vector<float>  stage2_one_bit_dists;

    rank_all_tokens_1bit_gpu(actual_k_stage1, k_rank_all_tokens,
                             stage2_doc_ids, stage2_one_bit_dists,
                             ws_.stream_compute);

#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.event_stage2_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventSynchronize(ws_.event_stage2_end));
    }
#endif

#ifdef CHIMERA_PROFILE
    auto stage3_wall_start = std::chrono::high_resolution_clock::now();
#endif

    rank_all_tokens_exbits_cpu(query_objs.data(),
                               stage2_doc_ids, stage2_one_bit_dists,
                               k, result);

#ifdef CHIMERA_PROFILE
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
        float s1_sum = s1_cagra + s1_expansion + s1_binary_ip + s1_atomic_agg + s1_sum_scores + s1_topk_sort + s1_d2d;

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

        if (profile_ptr != nullptr) {
            profile_ptr->persistent_stage23 = false;
            profile_ptr->total_search_time_ms = total_time;
            profile_ptr->stage1_time_ms = stage1_time;
            profile_ptr->s1_cagra_ms = s1_cagra;
            profile_ptr->s1_expansion_ms = s1_expansion;
            profile_ptr->s1_binary_ip_ms = s1_binary_ip;
            profile_ptr->s1_atomic_agg_ms = s1_atomic_agg;
            profile_ptr->s1_sum_scores_ms = s1_sum_scores;
            profile_ptr->s1_topk_sort_ms = s1_topk_sort;
            profile_ptr->s1_d2d_ms = s1_d2d;
            profile_ptr->s1_memset_ms = s1_memset;
            profile_ptr->s1_sum_accounted_ms = s1_sum;
            profile_ptr->stage2_time_ms = stage2_time;
            profile_ptr->stage3_time_ms = stage3_time_ms;
            profile_ptr->stage3_docs = static_cast<double>(stage2_doc_ids.size());
            profile_ptr->total_h2d_ms = total_h2d_ms;
            profile_ptr->total_d2h_ms = total_d2h_ms;
            profile_ptr->total_transfer_ms = total_h2d_ms + total_d2h_ms;
            profile_ptr->total_h2d_bytes = static_cast<double>(total_h2d_bytes);
            profile_ptr->total_d2h_bytes = static_cast<double>(total_d2h_bytes);
            profile_ptr->total_transfer_bytes =
                static_cast<double>(total_h2d_bytes + total_d2h_bytes);
        }
    }
#endif

#endif  // CHIMERA_OVERLAP_STAGE23

    auto search_end = std::chrono::high_resolution_clock::now();
    if constexpr (kProfile) {
        float search_wall_ms = std::chrono::duration<float, std::milli>(search_end - search_start).count();
        if (profile_ptr != nullptr) {
            profile_ptr->search_wall_ms = search_wall_ms;
        }
        if (print_profile) {
            gpu_search_profile_v6 profile_to_print =
                (profile_ptr != nullptr) ? *profile_ptr : gpu_search_profile_v6{};
            profile_to_print.search_wall_ms = search_wall_ms;
            print_gpu_search_profile_v6(profile_to_print, "[PROFILE]");
        }
    }

    return result;
}

// ======================== STAGE 1: GPU ========================

void chimera_index::rank_cluster_dists_gpu(
    query_object* h_query_objs,
    size_t nprobe, size_t k,
    int& actual_k_out,
    cudaStream_t stream
) {
    rank_cluster_dists_gpu_impl<false>(h_query_objs, nprobe, k, actual_k_out, stream);
}

template <bool kProfile>
void chimera_index::rank_cluster_dists_gpu_impl(
    query_object* h_query_objs,
    size_t nprobe, size_t k,
    int& actual_k_out,
    cudaStream_t stream
) {
#ifdef CHIMERA_COMPACT_DOC_BUFFER
    size_t doc_matrix_size = 0;
#else
    size_t doc_matrix_size = ws_.estimated_num_docs * Q_DOCLEN;
#endif
#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_memset_start, ws_.stream_d2h));
    }
#endif
#ifdef CHIMERA_COMPACT_DOC_BUFFER
    CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_bitmap, 0,
                               ws_.doc_bitmap_bucket_count * sizeof(doc_bitmap_bucket_t),
                               ws_.stream_d2h));
    CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_bitmap_offsets, 0,
                               ws_.doc_bitmap_offset_count * sizeof(doc_bitmap_offset_t),
                               ws_.stream_d2h));
#else
    CUDA_CHECK(cudaMemsetAsync(
        ws_.d_doc_query_max,
        0,
        doc_matrix_size * sizeof(float),
        ws_.stream_d2h));
    CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_touched, 0, ws_.estimated_num_docs * sizeof(int), ws_.stream_d2h));
#endif
    CUDA_CHECK(cudaMemsetAsync(ws_.d_num_unique_docs, 0, sizeof(int), ws_.stream_d2h));
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

    ivf->search_batch_gpu(ws_.d_queries, Q_DOCLEN, nprobe,
                          ws_.d_cagra_dists, ws_.d_cagra_labels, stream,
                          static_cast<size_t>(itopk_size));

#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_cagra_end, stream));
        CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_start, stream));
    }
#endif

#ifdef CHIMERA_USE_LUT
    // LUT path: the kernel iterates clusters directly from CAGRA labels,
    // so we skip the expansion kernel and d_emb_ids entirely. This removes
    // the emb_id load → code load dependency chain (400 cycles savings)
    // and the expansion kernel itself (~0.18 ms).
#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_end, stream));
    }
#endif
#else
    // Non-LUT path still needs the expansion.
    compute_query_expansion_sizes_kernel<<<(Q_DOCLEN + 255) / 256, 256, 0, stream>>>(
        ws_.d_cagra_labels,
        d_cluster_pos_,
        ws_.d_pair_offsets + 1,
        nprobe,
        ivf->n_clusters,
        Q_DOCLEN
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
        Q_DOCLEN,
        false
    );

#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_end, stream));
    }
#endif
#endif  // CHIMERA_USE_LUT

    int threads_per_block = 256;

    // The bitmap/doc-query buffers are cleared on stream_d2h while CAGRA
    // runs, so waiting here is free on the critical path.
    CUDA_CHECK(cudaStreamWaitEvent(stream, ws_.event_h2d_done));

#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_start, stream));
    }
#endif
    int h_num_touched = 0;
#ifdef CHIMERA_USE_LUT
#if defined(CHIMERA_CLUSTERED_LAYOUT_DISABLED)
    {
        int blocks_x = std::max(
            (max_cluster_size + threads_per_block - 1) / threads_per_block,
            16);
        dim3 grid(blocks_x, Q_DOCLEN);

        stage1_binary_ip_lut_nonclustered_flag_docs_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_cagra_labels, d_cluster_pos_,
            d_doc_ids_,
            d_doc_ptrs_,
            d_doc_block_lut_,
            d_inv_list_,
            ws_.d_doc_bitmap,
            num_docs,
            nprobe,
            ivf->n_clusters
        );
        CUDA_CHECK(cudaGetLastError());

        bitmap_offset_init_kernel<<<
            (ws_.doc_bitmap_bucket_count + threads_per_block - 1) / threads_per_block,
            threads_per_block,
            0,
            stream>>>(
                ws_.d_doc_bitmap,
                ws_.doc_bitmap_bucket_count,
                ws_.d_doc_bitmap_offsets);
        CUDA_CHECK(cudaGetLastError());

        size_t scan_temp = ws_.cub_temp_storage_bytes;
        cub::DeviceScan::ExclusiveSum(
            ws_.d_cub_temp_storage,
            scan_temp,
            ws_.d_doc_bitmap_offsets,
            ws_.d_doc_bitmap_offsets,
            ws_.doc_bitmap_offset_count,
            stream);

        doc_bitmap_offset_t h_num_touched_u32 = 0;
        CUDA_CHECK(cudaMemcpyAsync(
            &h_num_touched_u32,
            ws_.d_doc_bitmap_offsets + ws_.doc_bitmap_offset_count - 1,
            sizeof(doc_bitmap_offset_t),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        h_num_touched = static_cast<int>(h_num_touched_u32);
        if (h_num_touched == 0) {
            actual_k_out = 0;
            return;
        }
        ensure_compact_doc_capacity(static_cast<size_t>(h_num_touched));
        CUDA_CHECK(cudaMemsetAsync(
            ws_.d_doc_query_max, 0,
            static_cast<size_t>(h_num_touched) * Q_DOCLEN * sizeof(float),
            stream));

        bitmap_unique_docs_kernel<<<
            (ws_.doc_bitmap_bucket_count + threads_per_block - 1) / threads_per_block,
            threads_per_block,
            0,
            stream>>>(
                ws_.d_doc_bitmap,
                ws_.doc_bitmap_bucket_count,
                ws_.d_doc_bitmap_offsets,
                ws_.d_unique_doc_ids);
        CUDA_CHECK(cudaGetLastError());

        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_start, stream));
        }

        if (d_doc_ids_ != nullptr) {
            stage1_binary_ip_lut_nonclustered_kernel<<<grid, threads_per_block, 0, stream>>>(
                ws_.d_lut, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
                ws_.d_cagra_labels, d_cluster_pos_,
                d_doc_ids_,
                d_inv_list_,
                ws_.d_doc_query_max,
#ifdef CHIMERA_COMPACT_DOC_BUFFER
                ws_.d_doc_bitmap,
                ws_.d_doc_bitmap_offsets,
                h_num_touched,
#else
                ws_.d_doc_touched,
                ws_.d_unique_doc_ids,
                ws_.d_num_unique_docs,
#endif
                num_docs,
                nprobe,
                ivf->n_clusters
            );
        } else {
            stage1_binary_ip_lut_nonclustered_docptr_kernel<<<grid, threads_per_block, 0, stream>>>(
                ws_.d_lut, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
                ws_.d_cagra_labels, d_cluster_pos_,
                d_doc_ptrs_,
                d_doc_block_lut_,
                d_inv_list_,
                ws_.d_doc_query_max,
                ws_.d_doc_bitmap,
                ws_.d_doc_bitmap_offsets,
                h_num_touched,
                num_docs,
                nprobe,
                ivf->n_clusters
            );
        }
        CUDA_CHECK(cudaGetLastError());

        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_end, stream));
            CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_end, stream));
        }

        actual_k_out = std::min((int)k, h_num_touched);
    }
#else
    if (use_clustered_) {
        // Clustered layout: flat iteration across all probed clusters with
        // coalesced memory access. Ensure enough blocks for good SM occupancy.
        int blocks_x = std::max(
            (max_cluster_size + threads_per_block - 1) / threads_per_block,
            16);
        dim3 grid(blocks_x, Q_DOCLEN);

        stage1_binary_ip_lut_flag_docs_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_cagra_labels, d_cluster_pos_,
            d_clustered_doc_ids_,
            ws_.d_doc_bitmap,
            num_docs,
            nprobe,
            ivf->n_clusters
        );
        CUDA_CHECK(cudaGetLastError());

        bitmap_offset_init_kernel<<<
            (ws_.doc_bitmap_bucket_count + threads_per_block - 1) / threads_per_block,
            threads_per_block,
            0,
            stream>>>(
                ws_.d_doc_bitmap,
                ws_.doc_bitmap_bucket_count,
                ws_.d_doc_bitmap_offsets);
        CUDA_CHECK(cudaGetLastError());

        size_t scan_temp = ws_.cub_temp_storage_bytes;
        cub::DeviceScan::ExclusiveSum(
            ws_.d_cub_temp_storage,
            scan_temp,
            ws_.d_doc_bitmap_offsets,
            ws_.d_doc_bitmap_offsets,
            ws_.doc_bitmap_offset_count,
            stream);

        doc_bitmap_offset_t h_num_touched_u32 = 0;
        CUDA_CHECK(cudaMemcpyAsync(
            &h_num_touched_u32,
            ws_.d_doc_bitmap_offsets + ws_.doc_bitmap_offset_count - 1,
            sizeof(doc_bitmap_offset_t),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        h_num_touched = static_cast<int>(h_num_touched_u32);
        if (h_num_touched == 0) {
            actual_k_out = 0;
            return;
        }
        ensure_compact_doc_capacity(static_cast<size_t>(h_num_touched));
        CUDA_CHECK(cudaMemsetAsync(
            ws_.d_doc_query_max, 0,
            static_cast<size_t>(h_num_touched) * Q_DOCLEN * sizeof(float),
            stream));

        bitmap_unique_docs_kernel<<<
            (ws_.doc_bitmap_bucket_count + threads_per_block - 1) / threads_per_block,
            threads_per_block,
            0,
            stream>>>(
                ws_.d_doc_bitmap,
                ws_.doc_bitmap_bucket_count,
                ws_.d_doc_bitmap_offsets,
                ws_.d_unique_doc_ids);
        CUDA_CHECK(cudaGetLastError());

        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_start, stream));
        }

        stage1_binary_ip_lut_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_lut, d_clustered_code_, d_clustered_factor_, ws_.d_cb1_sumq,
            ws_.d_cagra_labels, d_cluster_pos_,
            d_clustered_doc_ids_,
            ws_.d_doc_query_max,
            ws_.d_doc_bitmap,
            ws_.d_doc_bitmap_offsets,
            h_num_touched,
            num_docs,
            nprobe,
            ivf->n_clusters
        );
        CUDA_CHECK(cudaGetLastError());

        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_end, stream));
            CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_end, stream));
        }

        actual_k_out = std::min((int)k, h_num_touched);
    } else {
        // Non-clustered fallback: explicit cluster iteration with inv_list
        // indirection, smem tiling, and __ldg() for scattered reads.
        int blocks_x = std::max(
            (max_cluster_size + threads_per_block - 1) / threads_per_block,
            16);
        dim3 grid(blocks_x, Q_DOCLEN);

        stage1_binary_ip_lut_nonclustered_flag_docs_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_cagra_labels, d_cluster_pos_,
            d_doc_ids_,
            d_doc_ptrs_,
            d_doc_block_lut_,
            d_inv_list_,
            ws_.d_doc_bitmap,
            num_docs,
            nprobe,
            ivf->n_clusters
        );
        CUDA_CHECK(cudaGetLastError());

        bitmap_offset_init_kernel<<<
            (ws_.doc_bitmap_bucket_count + threads_per_block - 1) / threads_per_block,
            threads_per_block,
            0,
            stream>>>(
                ws_.d_doc_bitmap,
                ws_.doc_bitmap_bucket_count,
                ws_.d_doc_bitmap_offsets);
        CUDA_CHECK(cudaGetLastError());

        size_t scan_temp = ws_.cub_temp_storage_bytes;
        cub::DeviceScan::ExclusiveSum(
            ws_.d_cub_temp_storage,
            scan_temp,
            ws_.d_doc_bitmap_offsets,
            ws_.d_doc_bitmap_offsets,
            ws_.doc_bitmap_offset_count,
            stream);

        doc_bitmap_offset_t h_num_touched_u32 = 0;
        CUDA_CHECK(cudaMemcpyAsync(
            &h_num_touched_u32,
            ws_.d_doc_bitmap_offsets + ws_.doc_bitmap_offset_count - 1,
            sizeof(doc_bitmap_offset_t),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        h_num_touched = static_cast<int>(h_num_touched_u32);
        if (h_num_touched == 0) {
            actual_k_out = 0;
            return;
        }
        ensure_compact_doc_capacity(static_cast<size_t>(h_num_touched));
        CUDA_CHECK(cudaMemsetAsync(
            ws_.d_doc_query_max, 0,
            static_cast<size_t>(h_num_touched) * Q_DOCLEN * sizeof(float),
            stream));

        bitmap_unique_docs_kernel<<<
            (ws_.doc_bitmap_bucket_count + threads_per_block - 1) / threads_per_block,
            threads_per_block,
            0,
            stream>>>(
                ws_.d_doc_bitmap,
                ws_.doc_bitmap_bucket_count,
                ws_.d_doc_bitmap_offsets,
                ws_.d_unique_doc_ids);
        CUDA_CHECK(cudaGetLastError());

        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_start, stream));
        }

        if (d_doc_ids_ != nullptr) {
            stage1_binary_ip_lut_nonclustered_kernel<<<grid, threads_per_block, 0, stream>>>(
                ws_.d_lut, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
                ws_.d_cagra_labels, d_cluster_pos_,
                d_doc_ids_,
                d_inv_list_,
                ws_.d_doc_query_max,
                ws_.d_doc_bitmap,
                ws_.d_doc_bitmap_offsets,
                h_num_touched,
                num_docs,
                nprobe,
                ivf->n_clusters
            );
        } else {
            stage1_binary_ip_lut_nonclustered_docptr_kernel<<<grid, threads_per_block, 0, stream>>>(
                ws_.d_lut, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
                ws_.d_cagra_labels, d_cluster_pos_,
                d_doc_ptrs_,
                d_doc_block_lut_,
                d_inv_list_,
                ws_.d_doc_query_max,
                ws_.d_doc_bitmap,
                ws_.d_doc_bitmap_offsets,
                h_num_touched,
                num_docs,
                nprobe,
                ivf->n_clusters
            );
        }
        CUDA_CHECK(cudaGetLastError());

        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_end, stream));
            CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_end, stream));
        }

        actual_k_out = std::min((int)k, h_num_touched);
    }
#endif
#else
    {
        int max_embs_per_query = ws_.max_embs_per_query_bound;
        int blocks_x = (max_embs_per_query + threads_per_block - 1) / threads_per_block;
        dim3 grid(blocks_x, Q_DOCLEN);

        stage1_binary_ip_kernel_v2<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_queries, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
            ws_.d_emb_ids, ws_.d_pair_offsets, ws_.d_emb_dists,
            max_embs_per_query
        );
        CUDA_CHECK(cudaGetLastError());

        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_end, stream));
            CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_start, stream));
        }

        aggregate_stage1_tracked_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_emb_ids,
            ws_.d_emb_dists,
            ws_.d_pair_offsets,
            d_doc_ids_,
            ws_.d_doc_query_max,
            ws_.d_doc_touched,
            ws_.d_unique_doc_ids,
            ws_.d_num_unique_docs,
            num_docs,
            ws_.max_stage1_pairs,
            max_embs_per_query
        );
        CUDA_CHECK(cudaGetLastError());

        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_end, stream));
        }

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
    }
#endif
    CUDA_CHECK(cudaGetLastError());

    const int thread_count = SUM_SCORES_WARPS_PER_BLOCK * 32; // 256
    if (h_num_touched == 0) {
        return;
    }
    int sparse_blocks =
        (h_num_touched + SUM_SCORES_WARPS_PER_BLOCK - 1) /
        SUM_SCORES_WARPS_PER_BLOCK;
#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_sum_scores_start, stream));
    }
#endif
    sum_doc_scores_sparse_kernel<<<sparse_blocks, thread_count, 0, stream>>>(
        ws_.d_doc_query_max,
        ws_.d_unique_doc_ids,
        ws_.d_stage1_doc_scores,
        ws_.d_topk_doc_ids,
#ifdef CHIMERA_COMPACT_DOC_BUFFER
        h_num_touched
#else
        h_num_touched,
        num_docs
#endif
    );
    CUDA_CHECK(cudaGetLastError());
#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_sum_scores_end, stream));
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
#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_topk_sort_end, stream));
        CUDA_CHECK(cudaEventRecord(ws_.s1_d2d_start, stream));
    }
#endif
    CUDA_CHECK(cudaMemcpyAsync(ws_.d_topk_doc_ids, ws_.d_sorted_doc_ids,
                               actual_k_out * sizeof(int), cudaMemcpyDeviceToDevice, stream));
#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_d2d_end, stream));
    }
#endif
}

// ======================== rank_stage23_persistent ========================

void chimera_index::rank_stage23_persistent(
    int num_candidates,
    size_t k,
    size_t k_stage2,
    query_object* queries,
    std::vector<size_t>& result
) {
    rank_stage23_persistent_impl<false>(
        num_candidates, k, k_stage2, queries, result, nullptr, true);
}

template <bool kProfile>
void chimera_index::rank_stage23_persistent_impl(
    int num_candidates,
    size_t k,
    size_t k_stage2,
    query_object* queries,
    std::vector<size_t>& result,
    gpu_search_profile_v6* profile,
    bool print_profile
) {
    (void)print_profile;
    if (num_candidates == 0) return;

    auto cpu_ms = [](auto a, auto b) { return std::chrono::duration<float, std::milli>(b - a).count(); };

    cudaStream_t stream = ws_.stream_compute;
    cudaStream_t stream_d2h = ws_.stream_d2h;
    int actual_k = (int)std::min(k_stage2, (size_t)num_candidates);

#ifdef CHIMERA_TIMELINE
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t tl_base;
    CUDA_CHECK(cudaEventCreate(&tl_base));
    CUDA_CHECK(cudaEventRecord(tl_base, stream));
    CUDA_CHECK(cudaEventSynchronize(tl_base));
    auto tl_base_cpu = std::chrono::high_resolution_clock::now();

    struct TLSpan { int start_us, end_us; std::string name, type; };
    std::vector<TLSpan> tl_gpu_compute, tl_gpu_d2h, tl_gpu_extract, tl_cpu;
    int tl_actual_chunks = 0;

    auto tl_cpu_us = [&](std::chrono::high_resolution_clock::time_point tp) -> int {
        return (int)std::chrono::duration_cast<std::chrono::microseconds>(tp - tl_base_cpu).count();
    };
    auto tl_gpu_us = [&](cudaEvent_t e) -> int {
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, tl_base, e)); return (int)(ms * 1000.0f);
    };

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

    // Phase A: Data Preparation (GPU)
    auto t_start_phase_a = std::chrono::high_resolution_clock::now();
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.phase_a_start_event, stream));
    }
    float time_phase_a_gather = 0;
    float time_phase_a_prefix = 0;
    float time_phase_a_token_ids = 0;
    float time_phase_a_d2h = 0;

    auto t_cpu0 = std::chrono::high_resolution_clock::now();

    int threads = 256;
    int blocks = (num_candidates + threads - 1) / threads;

    gather_doc_lengths_kernel<<<blocks, threads, 0, stream>>>(
        ws_.d_topk_doc_ids, d_doc_ptrs_,
        ws_.d_pair_doc_ids,
        num_candidates
    );
    CUDA_CHECK(cudaGetLastError());
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.phase_a_gather_done_event, stream));
    }

    auto t_cpu1 = std::chrono::high_resolution_clock::now();

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
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.phase_a_prefix_done_event, stream));
    }

    auto t_cpu2 = std::chrono::high_resolution_clock::now();

    gather_token_ids_kernel<<<num_candidates, 256, 0, stream>>>(
        ws_.d_topk_doc_ids,
        d_doc_ptrs_,
        ws_.d_pst_candidate_offsets,
        ws_.d_pst_token_ids,
        num_candidates
    );
    CUDA_CHECK(cudaGetLastError());
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.phase_a_token_ids_done_event, stream));
    }

    auto t_cpu3 = std::chrono::high_resolution_clock::now();

    size_t total_tokens;
    if constexpr (kProfile) {
        XFER_RECORD_BEGIN(stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_pst_total_tokens, ws_.d_pst_candidate_offsets + num_candidates,
                                sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_pst_candidate_offsets, ws_.d_pst_candidate_offsets,
                                (num_candidates + 1) * sizeof(size_t),
                                cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_pst_candidate_doc_ids, ws_.d_topk_doc_ids,
                                num_candidates * sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    if constexpr (kProfile) {
        XFER_RECORD_END(stream, sizeof(size_t) + (num_candidates + 1) * sizeof(size_t) + num_candidates * sizeof(int), false);
        CUDA_CHECK(cudaEventRecord(ws_.phase_a_d2h_done_event, stream));
    }

    auto t_cpu4 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaStreamSynchronize(stream));
    total_tokens = *ws_.h_pinned_pst_total_tokens;
    auto* h_candidate_doc_ids = ws_.h_pinned_pst_candidate_doc_ids;

    auto t_cpu5 = std::chrono::high_resolution_clock::now();

    float time_phase_a_gpu = 0;
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventElapsedTime(&time_phase_a_gather, ws_.phase_a_start_event, ws_.phase_a_gather_done_event));
        CUDA_CHECK(cudaEventElapsedTime(&time_phase_a_prefix, ws_.phase_a_gather_done_event, ws_.phase_a_prefix_done_event));
        CUDA_CHECK(cudaEventElapsedTime(&time_phase_a_token_ids, ws_.phase_a_prefix_done_event, ws_.phase_a_token_ids_done_event));
        CUDA_CHECK(cudaEventElapsedTime(&time_phase_a_d2h, ws_.phase_a_token_ids_done_event, ws_.phase_a_d2h_done_event));
        CUDA_CHECK(cudaEventElapsedTime(&time_phase_a_gpu, ws_.phase_a_start_event, ws_.phase_a_d2h_done_event));
    }

    auto t_cpu6 = std::chrono::high_resolution_clock::now();
    float time_phase_a = std::chrono::duration<float, std::milli>(t_cpu6 - t_start_phase_a).count();

    // Phase B: Launch ALL chunk GPU kernels
    auto t_start_phase_b = std::chrono::high_resolution_clock::now();
    int score_threads = 128;
    while (score_threads < Q_DOCLEN && score_threads < 256) score_threads *= 2;
    score_threads = std::min(score_threads, 256);

    int actual_chunks = std::min(std::min(overlap_chunks, N_OVERLAP_CHUNKS), num_candidates);
    int cand_chunk_size = (num_candidates + actual_chunks - 1) / actual_chunks;
#ifdef CHIMERA_TIMELINE
    tl_actual_chunks = actual_chunks;
#endif

    cudaEvent_t chunk_compute_done[N_OVERLAP_CHUNKS];
    cudaEvent_t chunk_d2h_done[N_OVERLAP_CHUNKS];
    cudaEvent_t chunk_extract_done[N_OVERLAP_CHUNKS];
    for (int c = 0; c < actual_chunks; c++) {
        CUDA_CHECK(cudaEventCreateWithFlags(&chunk_compute_done[c], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&chunk_d2h_done[c], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&chunk_extract_done[c], cudaEventDisableTiming));
    }

#ifdef CHIMERA_PROFILE
    cudaEvent_t chunk_bip_start[N_OVERLAP_CHUNKS];
    cudaEvent_t chunk_bip_end[N_OVERLAP_CHUNKS];
    cudaEvent_t chunk_docscore_start[N_OVERLAP_CHUNKS];
    cudaEvent_t chunk_docscore_end[N_OVERLAP_CHUNKS];
    bool chunk_has_bip[N_OVERLAP_CHUNKS] = {};
    if constexpr (kProfile) {
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
    }
#endif

    auto launch_chunk_compute = [&](int c) {
        int c_start = c * cand_chunk_size;
        int c_end = std::min(c_start + cand_chunk_size, num_candidates);
        int c_count = c_end - c_start;
        size_t tok_start = ws_.h_pinned_pst_candidate_offsets[c_start];
        size_t tok_end = ws_.h_pinned_pst_candidate_offsets[c_end];
        size_t tok_count = tok_end - tok_start;

        if (tok_count > 0) {
            int bip_blocks = (tok_count + 255) / 256;
#ifdef CHIMERA_PROFILE
            if constexpr (kProfile) {
                chunk_has_bip[c] = true;
                CUDA_CHECK(cudaEventRecord(chunk_bip_start[c], stream));
            }
#endif
#ifdef CHIMERA_USE_LUT
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
#ifdef CHIMERA_PROFILE
            if constexpr (kProfile) {
                CUDA_CHECK(cudaEventRecord(chunk_bip_end[c], stream));
            }
#endif
            CUDA_CHECK(cudaGetLastError());
        }

#ifdef CHIMERA_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(chunk_docscore_start[c], stream));
        }
#endif
        doc_score_kernel<<<c_count, score_threads, 0, stream>>>(
            ws_.d_token_dists, ws_.d_pst_candidate_offsets + c_start,
            ws_.d_doc_scores + c_start, total_tokens, c_count
        );
#ifdef CHIMERA_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(chunk_docscore_end[c], stream));
        }
#endif
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(chunk_compute_done[c], stream));
#ifdef CHIMERA_PROFILE
        if constexpr (kProfile) {
            if (c == actual_chunks - 1)
                CUDA_CHECK(cudaEventRecord(ws_.s23_pst_kernel_end, stream));
        }
#endif

        int c_start_d2h = c * cand_chunk_size;
        int c_end_d2h = std::min(c_start_d2h + cand_chunk_size, num_candidates);
        int c_count_d2h = c_end_d2h - c_start_d2h;
#ifdef CHIMERA_TIMELINE
        CUDA_CHECK(cudaEventRecord(tl_c_wait_ev[c], stream_d2h));
#endif
        CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, chunk_compute_done[c], 0));
#ifdef CHIMERA_TIMELINE
        CUDA_CHECK(cudaEventRecord(tl_c_scores_s[c], stream_d2h));
        tl_c_has_scores[c] = true;
#endif
        if constexpr (kProfile) {
            XFER_RECORD_BEGIN(stream_d2h);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            ws_.h_mapped_doc_scores + c_start_d2h,
            ws_.d_doc_scores + c_start_d2h,
            c_count_d2h * sizeof(float),
            cudaMemcpyDeviceToHost, stream_d2h));
        if constexpr (kProfile) {
            XFER_RECORD_END(stream_d2h, c_count_d2h * sizeof(float), false);
        }
#ifdef CHIMERA_TIMELINE
        CUDA_CHECK(cudaEventRecord(tl_c_scores_e[c], stream_d2h));
#endif
        CUDA_CHECK(cudaEventRecord(chunk_d2h_done[c], stream_d2h));
        CUDA_CHECK(cudaEventRecord(chunk_extract_done[c], stream_d2h));
    };

    for (int c = 0; c < actual_chunks; c++) {
        launch_chunk_compute(c);
    }
    CUDA_CHECK(cudaEventRecord(ws_.pst_extract_done, stream));
    auto t_end_phase_b = std::chrono::high_resolution_clock::now();
    float time_phase_b = std::chrono::duration<float, std::milli>(t_end_phase_b - t_start_phase_b).count();

    // Phase C: Streaming CPU processing
    auto t_start_phase_c = std::chrono::high_resolution_clock::now();
    double time_wait_d2h = 0;
    double time_running_topk = 0;
    double time_identify_new = 0;
    double time_gpu_extract = 0;
    double time_cpu_ip_ex = 0;
    double time_wait_extract = 0;
    double time_combine = 0;

    std::priority_queue<std::pair<float, size_t>> cpu_heap;
    auto& seen_doc_map = ws_.seen_doc_map;
    std::vector<int> seen_doc_list;
    seen_doc_list.reserve(actual_k * 2);
    int total_refined = 0;
    int prev_run_k = 0;

    auto& running_indices = ws_.running_indices;
    auto& h_sel_indices   = ws_.h_sel_indices;
    auto& h_out_offsets   = ws_.h_out_offsets;
    auto& ip_ex_buf       = ws_.ip_ex_buf;
    auto& refined_scores  = ws_.refined_scores;

    #pragma omp parallel for schedule(static)
    for (int _i = 0; _i < 1; _i++) { (void)_i; }

    for (int c = 0; c < actual_chunks; c++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int c_start = c * cand_chunk_size;
        int c_end = std::min(c_start + cand_chunk_size, num_candidates);
        int c_count = c_end - c_start;
        CUDA_CHECK(cudaEventSynchronize(chunk_d2h_done[c]));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_wait_d2h += std::chrono::duration<double, std::milli>(t1 - t0).count();
#ifdef CHIMERA_TIMELINE
        tl_cpu.push_back({tl_cpu_us(t0), tl_cpu_us(t1), "wait_d2h_c" + std::to_string(c), "async"});
#endif

        std::iota(running_indices.begin() + prev_run_k,
                  running_indices.begin() + prev_run_k + c_count, c_start);
        int merge_size = prev_run_k + c_count;
        int carry_k = std::min(actual_k, merge_size);
        const float* scores_ptr = ws_.h_mapped_doc_scores;
        auto cmp = [scores_ptr](int a, int b) {
            return scores_ptr[a] > scores_ptr[b];
        };
        std::nth_element(running_indices.begin(), running_indices.begin() + carry_k,
                        running_indices.begin() + merge_size, cmp);
        int run_k = std::min(actual_k * (c + 1) / actual_chunks, carry_k);
        std::nth_element(running_indices.begin(), running_indices.begin() + run_k,
                        running_indices.begin() + carry_k, cmp);
        prev_run_k = carry_k;
        auto t2 = std::chrono::high_resolution_clock::now();
        time_running_topk += std::chrono::duration<double, std::milli>(t2 - t1).count();
#ifdef CHIMERA_TIMELINE
        tl_cpu.push_back({tl_cpu_us(t1), tl_cpu_us(t2), "topk_c" + std::to_string(c), "async"});
#endif

        std::vector<std::pair<float, int>> new_docs;
        new_docs.reserve(run_k);
        for (int i = 0; i < run_k; i++) {
            int cand_idx = running_indices[i];
            int doc_id = h_candidate_doc_ids[cand_idx];
            if (!seen_doc_map[doc_id]) {
                new_docs.emplace_back(scores_ptr[cand_idx], cand_idx);
            }
        }

        if (new_docs.empty()) {
#ifdef CHIMERA_TIMELINE
            auto t_skip = std::chrono::high_resolution_clock::now();
            tl_cpu.push_back({tl_cpu_us(t2), tl_cpu_us(t_skip), "identify_c" + std::to_string(c), "async"});
#endif
            continue;
        }
        std::sort(new_docs.begin(), new_docs.end(), std::greater<>());
        int to_refine = new_docs.size();
        new_docs.resize(to_refine);
        for (const auto& p : new_docs) {
            int doc_id = h_candidate_doc_ids[p.second];
            seen_doc_map[doc_id] = true;
            seen_doc_list.push_back(doc_id);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        time_identify_new += std::chrono::duration<double, std::milli>(t3 - t2).count();
#ifdef CHIMERA_TIMELINE
        tl_cpu.push_back({tl_cpu_us(t2), tl_cpu_us(t3), "identify_c" + std::to_string(c), "async"});
#endif

        h_out_offsets[0] = 0;
        for (int i = 0; i < to_refine; i++) {
            h_sel_indices[i] = new_docs[i].second;
            int doc_id = h_candidate_doc_ids[h_sel_indices[i]];
            h_out_offsets[i + 1] = h_out_offsets[i] +
                (size_t)(doc_ptrs_[doc_id + 1] - doc_ptrs_[doc_id]);
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        time_gpu_extract += std::chrono::duration<double, std::milli>(t4 - t3).count();
#ifdef CHIMERA_TIMELINE
        tl_cpu.push_back({tl_cpu_us(t3), tl_cpu_us(t4), "launch_extract_c" + std::to_string(c), "async"});
#endif

        cpu_compute_ip_ex(
            h_candidate_doc_ids, h_sel_indices.data(), h_out_offsets.data(),
            to_refine, ws_.h_pinned_queries, ip_ex_buf.data());
        auto t5 = std::chrono::high_resolution_clock::now();
        time_cpu_ip_ex += std::chrono::duration<double, std::milli>(t5 - t4).count();
#ifdef CHIMERA_TIMELINE
        tl_cpu.push_back({tl_cpu_us(t4), tl_cpu_us(t5), "cpu_ip_ex_c" + std::to_string(c), "static"});
#endif

        CUDA_CHECK(cudaEventSynchronize(chunk_extract_done[c]));
        auto t6 = std::chrono::high_resolution_clock::now();
        time_wait_extract += std::chrono::duration<double, std::milli>(t6 - t5).count();
#ifdef CHIMERA_TIMELINE
        tl_cpu.push_back({tl_cpu_us(t5), tl_cpu_us(t6), "wait_extract_c" + std::to_string(c), "async"});
#endif

        cpu_combine_scores(
            h_candidate_doc_ids, h_sel_indices.data(), h_out_offsets.data(),
            to_refine, queries, ip_ex_buf.data(),
            refined_scores.data());

        for (int i = 0; i < to_refine; i++) {
            cpu_heap.emplace(refined_scores[i].first, (size_t)refined_scores[i].second);
        }
        auto t7 = std::chrono::high_resolution_clock::now();
        time_combine += std::chrono::duration<double, std::milli>(t7 - t6).count();
#ifdef CHIMERA_TIMELINE
        tl_cpu.push_back({tl_cpu_us(t6), tl_cpu_us(t7), "combine_c" + std::to_string(c), "subflow"});
#endif

        total_refined += to_refine;
    }

    auto t_end_phase_c = std::chrono::high_resolution_clock::now();
    double time_phase_c = std::chrono::duration<double, std::milli>(t_end_phase_c - t_start_phase_c).count();
    double total_wall_time = std::chrono::duration<double, std::milli>(t_end_phase_c - t_start_phase_a).count();

#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
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
        if (profile != nullptr) {
            profile->persistent_stage23 = true;
            profile->phase_a_wall_ms = time_phase_a;
            profile->phase_a_gpu_total_ms = time_phase_a_gpu;
            profile->phase_a_gather_ms = time_phase_a_gather;
            profile->phase_a_prefix_ms = time_phase_a_prefix;
            profile->phase_a_token_ids_ms = time_phase_a_token_ids;
            profile->phase_a_d2h_ms = time_phase_a_d2h;
            profile->phase_a_cpu_launch_gather_ms = cpu_ms(t_cpu0, t_cpu1);
            profile->phase_a_cpu_launch_prefix_ms = cpu_ms(t_cpu1, t_cpu2);
            profile->phase_a_cpu_launch_token_ids_ms = cpu_ms(t_cpu2, t_cpu3);
            profile->phase_a_cpu_launch_d2h_ms = cpu_ms(t_cpu3, t_cpu4);
            profile->phase_a_cpu_sync_ms = cpu_ms(t_cpu4, t_cpu5);
            profile->phase_a_cpu_event_elapsed_ms = cpu_ms(t_cpu5, t_cpu6);
            profile->phase_a_gpu_sum_accounted_ms =
                time_phase_a_gather + time_phase_a_prefix + time_phase_a_token_ids + time_phase_a_d2h;
            profile->phase_b_wall_ms = time_phase_b;
            profile->phase_c_wait_d2h_ms = time_wait_d2h;
            profile->phase_c_topk_ms = time_running_topk;
            profile->phase_c_identify_ms = time_identify_new;
            profile->phase_c_gpu_extract_ms = time_gpu_extract;
            profile->phase_c_cpu_ip_ex_ms = time_cpu_ip_ex;
            profile->phase_c_wait_extract_ms = time_wait_extract;
            profile->phase_c_combine_ms = time_combine;
            profile->phase_c_total_ms = time_phase_c;
            profile->phase_c_refined_docs = total_refined;
            profile->phase_abc_total_wall_ms = total_wall_time;
            profile->phase_b_binary_ip_total_ms = ws_.s23_pst_bip_ms;
            profile->phase_b_doc_score_total_ms = ws_.s23_pst_docscore_ms;
            profile->phase_b_total_kernel_ms = ws_.s23_pst_kernel_ms;
        }
    }
#endif

    // Phase D: Extract final top-k results
    result.clear();
    for (size_t i = 0; i < k && !cpu_heap.empty(); ++i) {
        result.push_back(cpu_heap.top().second);
        cpu_heap.pop();
    }

    for (int doc_id : seen_doc_list) seen_doc_map[doc_id] = false;

    for (int c = 0; c < actual_chunks; c++) {
        CUDA_CHECK(cudaEventDestroy(chunk_compute_done[c]));
        CUDA_CHECK(cudaEventDestroy(chunk_d2h_done[c]));
        CUDA_CHECK(cudaEventDestroy(chunk_extract_done[c]));
    }

#ifdef CHIMERA_TIMELINE
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamSynchronize(stream_d2h));
        CUDA_CHECK(cudaStreamSynchronize(ws_.stream_extract));

        tl_gpu_compute.push_back({tl_gpu_us(ws_.phase_a_start_event), tl_gpu_us(ws_.phase_a_gather_done_event), "gather_doc_lengths", "async"});
        tl_gpu_compute.push_back({tl_gpu_us(ws_.phase_a_gather_done_event), tl_gpu_us(ws_.phase_a_prefix_done_event), "prefix_sum", "async"});
        tl_gpu_compute.push_back({tl_gpu_us(ws_.phase_a_prefix_done_event), tl_gpu_us(ws_.phase_a_token_ids_done_event), "gather_token_ids", "async"});
        tl_gpu_compute.push_back({tl_gpu_us(ws_.phase_a_token_ids_done_event), tl_gpu_us(ws_.phase_a_d2h_done_event), "d2h_metadata", "async"});

#ifdef CHIMERA_PROFILE
        for (int c = 0; c < tl_actual_chunks; c++) {
            if (chunk_has_bip[c]) {
                tl_gpu_compute.push_back({tl_gpu_us(chunk_bip_start[c]), tl_gpu_us(chunk_bip_end[c]),
                    "binary_ip_c" + std::to_string(c), "static"});
            }
            tl_gpu_compute.push_back({tl_gpu_us(chunk_docscore_start[c]), tl_gpu_us(chunk_docscore_end[c]),
                "docscore_c" + std::to_string(c), "subflow"});
        }
#endif

        for (int c = 0; c < tl_actual_chunks; c++) {
            if (tl_c_has_scores[c]) {
                tl_gpu_d2h.push_back({tl_gpu_us(tl_c_wait_ev[c]), tl_gpu_us(tl_c_scores_s[c]),
                    "wait_compute_c" + std::to_string(c), "async"});
                tl_gpu_d2h.push_back({tl_gpu_us(tl_c_scores_s[c]), tl_gpu_us(tl_c_scores_e[c]),
                    "d2h_scores_c" + std::to_string(c), "condition"});
            }
            if (tl_c_has_extract[c]) {
                tl_gpu_extract.push_back({tl_gpu_us(tl_c_extract_s[c]), tl_gpu_us(tl_c_extract_e[c]),
                    "extract_v2_c" + std::to_string(c), "module"});
            }
        }

        tl_cpu.push_back({tl_cpu_us(t_cpu0), tl_cpu_us(t_cpu1), "launch_gather", "async"});
        tl_cpu.push_back({tl_cpu_us(t_cpu1), tl_cpu_us(t_cpu2), "launch_prefix", "async"});
        tl_cpu.push_back({tl_cpu_us(t_cpu2), tl_cpu_us(t_cpu3), "launch_token_ids", "async"});
        tl_cpu.push_back({tl_cpu_us(t_cpu3), tl_cpu_us(t_cpu4), "launch_d2h", "async"});
        tl_cpu.push_back({tl_cpu_us(t_cpu4), tl_cpu_us(t_cpu5), "sync_stream", "async"});
        tl_cpu.push_back({tl_cpu_us(t_cpu5), tl_cpu_us(t_cpu6), "event_elapsed", "async"});
        tl_cpu.push_back({tl_cpu_us(t_start_phase_b), tl_cpu_us(t_end_phase_b), "launch_phase_b", "async"});

        auto span_cmp = [](const TLSpan& a, const TLSpan& b) { return a.start_us < b.start_us; };
        std::sort(tl_gpu_compute.begin(), tl_gpu_compute.end(), span_cmp);
        std::sort(tl_gpu_d2h.begin(), tl_gpu_d2h.end(), span_cmp);
        std::sort(tl_gpu_extract.begin(), tl_gpu_extract.end(), span_cmp);
        std::sort(tl_cpu.begin(), tl_cpu.end(), span_cmp);

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
        oss << "{\"worker\":1,\"level\":0,\"data\":[";
        write_spans(oss, tl_gpu_compute);
        oss << "]},";
        oss << "{\"worker\":2,\"level\":0,\"data\":[";
        write_spans(oss, tl_gpu_d2h);
        oss << "]},";
        oss << "{\"worker\":3,\"level\":0,\"data\":[";
        write_spans(oss, tl_gpu_extract);
        oss << "]}]},";
        oss << "{\"executor\":1,\"data\":[";
        oss << "{\"worker\":1,\"level\":0,\"data\":[";
        write_spans(oss, tl_cpu);
        oss << "]}]}]";

        std::ofstream tlf("timeline.json");
        tlf << oss.str();
        tlf.close();
        std::cout << "[TIMELINE] Wrote timeline.json\n";

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

#ifdef CHIMERA_PROFILE
    if constexpr (kProfile) {
        for (int c = 0; c < actual_chunks; c++) {
            CUDA_CHECK(cudaEventDestroy(chunk_bip_start[c]));
            CUDA_CHECK(cudaEventDestroy(chunk_bip_end[c]));
            CUDA_CHECK(cudaEventDestroy(chunk_docscore_start[c]));
            CUDA_CHECK(cudaEventDestroy(chunk_docscore_end[c]));
        }
    }
#endif
}

// ======================== STAGE 3: CPU ========================

void chimera_index::cpu_compute_ip_ex(
    const int* h_candidate_doc_ids,
    const int* h_sel_indices,
    const size_t* h_out_offsets,
    int to_refine,
    const float* queries_flat,
    float* ip_ex_buf
) {
    const size_t full_code_stride = PADDED_DIM * (1 + ex_bits) / 8;
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < to_refine; i++) {
        int doc_id = h_candidate_doc_ids[h_sel_indices[i]];
        size_t doc_start = doc_ptrs_[doc_id];
        size_t n_tok = doc_ptrs_[doc_id + 1] - doc_start;
        alignas(64) float decoded[PADDED_DIM];
        for (size_t t = 0; t < n_tok; t++) {
            size_t tid = doc_start + t;
            unpack_func_(
                reinterpret_cast<const uint8_t*>(&full_code_[tid * full_code_stride]),
                decoded, PADDED_DIM);
            if (t + 1 < n_tok)
                __builtin_prefetch(&full_code_[(tid + 1) * full_code_stride], 0, 3);
            rabitqlib::gemv_batch8_avx512(
                queries_flat, decoded,
                &ip_ex_buf[(h_out_offsets[i] + t) * Q_DOCLEN],
                Q_DOCLEN, PADDED_DIM);
        }
    }
}

void chimera_index::cpu_combine_scores(
    const int* h_candidate_doc_ids,
    const int* h_sel_indices,
    const size_t* h_out_offsets,
    int to_refine,
    const query_object* queries,
    const float* ip_full_buf,
    std::pair<float, int>* refined_scores
) {
    alignas(64) float cbex_sumq_arr[Q_DOCLEN];
    for (size_t j = 0; j < Q_DOCLEN; j++)
        cbex_sumq_arr[j] = queries[j].cbex_sumq;

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < to_refine; i++) {
        int doc_id = h_candidate_doc_ids[h_sel_indices[i]];
        size_t doc_start = doc_ptrs_[doc_id];
        size_t n_tok = doc_ptrs_[doc_id + 1] - doc_start;

        alignas(64) float max_ts[Q_DOCLEN];
        __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        for (size_t j = 0; j < Q_DOCLEN; j += 16)
            _mm512_store_ps(&max_ts[j], neg_inf);

        for (size_t t = 0; t < n_tok; t++) {
            size_t tid = doc_start + t;
            float tok_exf = ex_factor_[tid];
            __m512 v_tok_exf = _mm512_set1_ps(tok_exf);

            const float* full_base = &ip_full_buf[(h_out_offsets[i] + t) * Q_DOCLEN];

            for (size_t j = 0; j < Q_DOCLEN; j += 16) {
                __m512 full_ip = _mm512_loadu_ps(&full_base[j]);
                __m512 cbex    = _mm512_load_ps(&cbex_sumq_arr[j]);
                __m512 combined = _mm512_mul_ps(
                    _mm512_sub_ps(full_ip, cbex),
                    v_tok_exf);
                _mm512_store_ps(&max_ts[j],
                    _mm512_max_ps(_mm512_load_ps(&max_ts[j]), combined));
            }
        }

        __m512 sum = _mm512_load_ps(&max_ts[0]);
        for (size_t j = 16; j < Q_DOCLEN; j += 16)
            sum = _mm512_add_ps(sum, _mm512_load_ps(&max_ts[j]));
        refined_scores[i] = {_mm512_reduce_add_ps(sum), doc_id};
    }
}

// ======================== DESTRUCTOR ========================

chimera_index::~chimera_index() {
    CUDA_CHECK(cudaDeviceSynchronize());

    // PG_CAGRA stores a RAFT resource whose active stream is rebound during search.
    // Release it before tearing down streams and other CUDA resources.
    delete ivf;
    ivf = nullptr;

    CUDA_CHECK(cudaFree(d_one_bit_code_));
    CUDA_CHECK(cudaFree(d_one_bit_factor_));
    CUDA_CHECK(cudaFree(d_doc_ids_));
    CUDA_CHECK(cudaFree(d_doc_ptrs_));
    CUDA_CHECK(cudaFree(d_doc_block_lut_));
    // d_inv_list_ may be allocated in non-LUT path or as fallback in LUT path;
    // cudaFree(nullptr) is a safe no-op
    CUDA_CHECK(cudaFree(d_inv_list_));
    CUDA_CHECK(cudaFree(d_cluster_pos_));

    // Clustered arrays are nullptr when fallback is active; cudaFree(nullptr) is safe
    CUDA_CHECK(cudaFree(d_clustered_code_));
    CUDA_CHECK(cudaFree(d_clustered_factor_));
    CUDA_CHECK(cudaFree(d_clustered_doc_ids_));

    CUDA_CHECK(cudaFree(ws_.d_queries));
    CUDA_CHECK(cudaFree(ws_.d_cb1_sumq));
#ifdef CHIMERA_USE_LUT
    CUDA_CHECK(cudaFree(ws_.d_lut));
#endif
    CUDA_CHECK(cudaFree(ws_.d_emb_ids));
    CUDA_CHECK(cudaFree(ws_.d_pair_offsets));
    CUDA_CHECK(cudaFree(ws_.d_emb_dists));
    CUDA_CHECK(cudaFree(ws_.d_pair_doc_ids));
#ifndef CHIMERA_OVERLAP_STAGE23
    CUDA_CHECK(cudaFree(ws_.d_candidate_offsets));
    CUDA_CHECK(cudaFree(ws_.d_token_dists));
    CUDA_CHECK(cudaFree(ws_.d_doc_scores));
#endif
    CUDA_CHECK(cudaFree(ws_.d_selected_indices));

    CUDA_CHECK(cudaFree(ws_.d_cagra_dists));
    CUDA_CHECK(cudaFree(ws_.d_cagra_labels));

    CUDA_CHECK(cudaFree(ws_.d_num_unique_docs));
#ifdef CHIMERA_COMPACT_DOC_BUFFER
    CUDA_CHECK(cudaFree(ws_.d_compact_doc_buffer_raw));
    CUDA_CHECK(cudaFree(ws_.d_doc_bitmap));
    CUDA_CHECK(cudaFree(ws_.d_doc_bitmap_offsets));
#else
    CUDA_CHECK(cudaFree(ws_.d_sorted_doc_ids));
    CUDA_CHECK(cudaFree(ws_.d_unique_doc_ids));
    CUDA_CHECK(cudaFree(ws_.d_stage1_doc_scores));
    CUDA_CHECK(cudaFree(ws_.d_doc_query_max));
    CUDA_CHECK(cudaFree(ws_.d_doc_touched));
    CUDA_CHECK(cudaFree(ws_.d_topk_scores));
    CUDA_CHECK(cudaFree(ws_.d_topk_doc_ids));
    CUDA_CHECK(cudaFree(ws_.d_topk_indices));
#endif
    CUDA_CHECK(cudaFree(ws_.d_cub_temp_storage));

    CUDA_CHECK(cudaFreeHost(ws_.h_pinned_queries));
    CUDA_CHECK(cudaFreeHost(ws_.h_pinned_cb1_sumq));
#ifndef CHIMERA_OVERLAP_STAGE23
    CUDA_CHECK(cudaFreeHost(ws_.h_pinned_batch_scores));
#endif

#ifdef CHIMERA_OVERLAP_STAGE23
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

    CUDA_CHECK(cudaStreamDestroy(ws_.stream_compute));
    CUDA_CHECK(cudaStreamDestroy(ws_.stream_h2d));
    CUDA_CHECK(cudaStreamDestroy(ws_.stream_d2h));
    CUDA_CHECK(cudaStreamDestroy(ws_.stream_extract));
    CUDA_CHECK(cudaEventDestroy(ws_.event_h2d_done));

#ifdef CHIMERA_PROFILE
    CUDA_CHECK(cudaEventDestroy(ws_.event_start));
    CUDA_CHECK(cudaEventDestroy(ws_.event_end));
    CUDA_CHECK(cudaEventDestroy(ws_.event_stage1_start));
    CUDA_CHECK(cudaEventDestroy(ws_.event_stage1_end));
    CUDA_CHECK(cudaEventDestroy(ws_.event_stage2_start));
    CUDA_CHECK(cudaEventDestroy(ws_.event_stage2_end));

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

    CUDA_CHECK(cudaEventDestroy(ws_.s23_pst_kernel_start));
    CUDA_CHECK(cudaEventDestroy(ws_.s23_pst_kernel_end));

    for (int i = 0; i < Workspace::MAX_XFER_RECORDS; i++) {
        CUDA_CHECK(cudaEventDestroy(ws_.xfer_records[i].start));
        CUDA_CHECK(cudaEventDestroy(ws_.xfer_records[i].end));
    }
#endif

    delete rotator_;
    rotator_ = nullptr;
}


}  // namespace Chimera
