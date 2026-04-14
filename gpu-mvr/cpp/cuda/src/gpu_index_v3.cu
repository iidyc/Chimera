#include "gpu_index_v3.cuh"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>

#include "clustered_stage1_file_format.hpp"
#include "gpu_index_layout.hpp"
#include "mvr_index_file_format.hpp"
#include "startup_profile.hpp"

// ======================== CONSTRUCTOR ========================

gpu_mvr_index::gpu_mvr_index(const std::string& filename, const std::vector<int>& doc_lens) {
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

    const size_t one_bit_bytes = n * (PADDED_DIM / 8);
    full_code_.resize(n * PADDED_DIM * (1 + ex_bits) / 8);
    ex_factor_.resize(n);

    inf.seekg(static_cast<std::streamoff>(one_bit_bytes), std::ios::cur);
    if (!inf) {
        throw std::runtime_error(
            "Failed to skip one-bit code section in " + resolved_paths.quantized_data_path);
    }
    inf.read(full_code_.data(), full_code_.size());
    inf.seekg(static_cast<std::streamoff>(n * sizeof(float)), std::ios::cur);
    if (!inf) {
        throw std::runtime_error(
            "Failed to skip one-bit factor section in " + resolved_paths.quantized_data_path);
    }
    inf.read((char*)ex_factor_.data(), n * sizeof(float));
    inf.close();
    startup.mark("read_quantized_payload_and_factors");

    ip_func_ = select_excode_ipfunc(1 + ex_bits);
    unpack_func_ = select_excode_unpackfunc(1 + ex_bits);

    ivf = new IVF_PG(n_clusters, d, PGType::CAGRA);
    if (resolved_paths.split_layout) {
        ivf->load(resolved_paths.ivf_path, resolved_paths.centroids_path);
    } else {
        ivf->load(filename);
    }
    max_cluster_size = ivf->max_cluster_size();
    std::cout << "max cluster size: " << max_cluster_size << "\n";
    startup.mark("load_ivf");

    set_doc_mapping(doc_lens);
    startup.mark("set_doc_mapping");

    size_t free_mem, total_mem;
    // CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // std::cout << "GPU memory usage after index upload: "
    //           << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB / "
    //           << (total_mem / (1024.0 * 1024.0)) << " MB\n";

    // size_t required_mem = n * PADDED_DIM / 8 + n * sizeof(float) * 2 + (num_docs + 1) * sizeof(int) +  // one-bit code + factor + doc mapping
    //                       (ivf->n_clusters + 1) * sizeof(size_t);  // cluster_pos
    // std::cout << "Estimated GPU memory required for index data: "
    //           << (required_mem / (1024.0 * 1024.0)) << " MB\n";
    // std::cout << "Free GPU memory: " << (free_mem / (1024.0 * 1024.0)) << " MB\n";

    // Allocate persistent GPU data
    CUDA_CHECK(cudaMalloc(&d_doc_ptrs_, (num_docs + 1) * sizeof(int)));

    // Upload inverted list structures to GPU
    CUDA_CHECK(cudaMalloc(&d_cluster_pos_, (ivf->n_clusters + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_cluster_pos_, ivf->cluster_pos.data(),
                          (ivf->n_clusters + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

    d_inv_list_ = nullptr;

    CUDA_CHECK(cudaMemcpy(d_doc_ptrs_, doc_ptrs_.data(), (num_docs + 1) * sizeof(int), cudaMemcpyHostToDevice));
    startup.mark("upload_persistent_data");

    // CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // std::cout << "GPU memory usage after index upload: "
    //           << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB / "
    //           << (total_mem / (1024.0 * 1024.0)) << " MB\n";

    allocate_workspace();
    startup.mark("allocate_workspace");

    // Load the persisted cluster-ordered GPU index. Stage-1 and stage-2 both
    // depend on this layout in v3.
    {
        size_t inv_n = ivf->inv_list.size();
        size_t code_per_vec = PADDED_DIM / 8;  // == CODE_BYTES
        size_t clustered_bytes =
            inv_n * (code_per_vec + sizeof(float) + sizeof(int) + sizeof(uint32_t));
        const auto& gpu_index_path = resolved_paths.gpu_index_path;

        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

        if (free_mem <= clustered_bytes) {
            throw std::runtime_error(
                "Insufficient GPU memory for required gpu_index.bin layout: need " +
                std::to_string(clustered_bytes) + " bytes, have " +
                std::to_string(free_mem) + " bytes free");
        }

        use_clustered_ = true;
        if (gpu_index_path.empty() || !std::filesystem::exists(gpu_index_path)) {
            throw std::runtime_error(
                "Missing gpu_index.bin for split GPU index layout in " +
                std::filesystem::path(resolved_paths.quantized_data_path).parent_path().string());
        }

        std::ifstream clustered_header_file(gpu_index_path, std::ios::binary);
        const auto clustered_header =
            clustered_stage1_file_format::read_header(
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
        const size_t expected_size =
            clustered_stage1_file_format::header_bytes() +
            inv_n * code_per_vec +
            inv_n * sizeof(float) +
            inv_n * sizeof(int) +
            inv_n * sizeof(uint32_t);
        if (mapping_size != expected_size) {
            close(clustered_fd);
            throw std::runtime_error(
                "Unexpected GPU index size for " + gpu_index_path);
        }

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
        const auto* clustered_code =
            clustered_base + clustered_stage1_file_format::header_bytes();
        const auto* clustered_factor = reinterpret_cast<const float*>(
            clustered_code + inv_n * code_per_vec);
        const auto* clustered_doc_ids = reinterpret_cast<const int*>(
            reinterpret_cast<const char*>(clustered_factor) + inv_n * sizeof(float));
        const auto* token_to_cluster_pos = reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const char*>(clustered_doc_ids) + inv_n * sizeof(int));

        CUDA_CHECK(cudaMalloc(&d_clustered_code_, inv_n * code_per_vec));
        CUDA_CHECK(cudaMalloc(&d_clustered_factor_, inv_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_clustered_doc_ids_, inv_n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_token_to_cluster_pos_, inv_n * sizeof(uint32_t)));

        CUDA_CHECK(cudaMemcpy(
            d_clustered_code_, clustered_code,
            inv_n * code_per_vec, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_clustered_factor_, clustered_factor,
            inv_n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_clustered_doc_ids_, clustered_doc_ids,
            inv_n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_token_to_cluster_pos_, token_to_cluster_pos,
            inv_n * sizeof(uint32_t), cudaMemcpyHostToDevice));

        munmap(clustered_mapping, mapping_size);
        close(clustered_fd);
        startup.note("gpu_index_source", "persisted");
        startup.mark("load_gpu_index");

        std::cout << "[gpu_mvr] Using cluster-ordered layout ("
                  << (clustered_bytes / (1024.0 * 1024.0)) << " MB)\n";
        startup.mark("build_clustered_layout");
    }
}

// ======================== set_doc_mapping ========================

void gpu_mvr_index::set_doc_mapping(const std::vector<int>& doc_lens) {
    num_docs = doc_lens.size();
    doc_ptrs_.resize(num_docs + 1, 0);
    for (size_t i = 0; i < num_docs; ++i) {
        max_doc_len = std::max(max_doc_len, doc_lens[i]);
        doc_ptrs_[i + 1] = doc_ptrs_[i] + doc_lens[i];
    }
}

// ======================== allocate_workspace ========================

void gpu_mvr_index::allocate_workspace() {
    ws_.max_q_doclen = Q_DOCLEN;
    // ws_.max_stage1_pairs = nprobe * max_cluster_size * Q_DOCLEN;
    ws_.max_stage1_pairs = nprobe * n / n_clusters * Q_DOCLEN;
    ws_.max_stage2_candidates = k_rank_cluster;
    ws_.max_stage2_tokens = ws_.max_stage2_candidates * max_doc_len;
    ws_.max_stage2_k = k_rank_all_tokens;
    ws_.max_stage2_k_tokens = ws_.max_stage2_k * max_doc_len;
    ws_.estimated_num_docs = (size_t)num_docs;

#ifdef GPU_MVR_COMPACT_DOC_BUFFER
    // Compact doc buffer: sized for the max docs that can be touched in
    // one query, NOT all docs. Upper bound = nprobe * max_cluster_size
    // (single query token). The union across Q_DOCLEN tokens shares many
    // clusters via CAGRA, so this is a safe over-estimate.
    // ws_.max_compact_docs = std::min((size_t)num_docs,
    //                                 (size_t)nprobe * max_cluster_size);
    ws_.max_compact_docs = std::min((size_t)num_docs,
                                    (size_t)nprobe * n / n_clusters * 32);
    // Hash table capacity: next power-of-2 >= 2× max_compact_docs (≤50% load).
    {
        size_t cap = 1;
        while (cap < 2 * ws_.max_compact_docs) cap <<= 1;
        ws_.ht_capacity = cap;
        ws_.ht_mask = (unsigned int)(cap - 1);
    }
    size_t doc_buf_rows = ws_.max_compact_docs;

    std::cout << "max cluster size: " << max_cluster_size << "\n";
    std::cout << "max_compact_docs: " << ws_.max_compact_docs
              << "  (num_docs=" << num_docs << ")\n";
    std::cout << "hash table capacity: " << ws_.ht_capacity << "\n";
#else
    size_t doc_buf_rows = ws_.estimated_num_docs;
    std::cout << "max cluster size: " << max_cluster_size << "\n";
#endif

    size_t estimated_size = Q_DOCLEN * PADDED_DIM * sizeof(float) +  // d_queries
                            Q_DOCLEN * sizeof(float) +  // d_cb1_sumq
                            ws_.max_stage1_pairs * sizeof(size_t) +  // d_emb_ids
                            (Q_DOCLEN + 1) * sizeof(int) +  // d_pair_offsets
                            ws_.max_stage1_pairs * sizeof(float) +  // d_emb_dists
                            ws_.max_stage1_pairs * sizeof(int) +  // d_pair_doc_ids
                            ws_.max_stage1_pairs * sizeof(int) +  // d_sorted_doc_ids
                            doc_buf_rows * sizeof(int) +  // d_unique_doc_ids
                            doc_buf_rows * sizeof(float) +  // d_stage1_doc_scores
                            doc_buf_rows * Q_DOCLEN * sizeof(float) +  // d_doc_query_max
                            sizeof(int) +  // d_num_unique_docs
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
                            ws_.ht_capacity * 2 * sizeof(int);  // d_ht_keys + d_ht_vals
#else
                            ws_.estimated_num_docs * sizeof(int);  // d_doc_touched
#endif
    std::cout << "Estimated GPU memory required for workspace: "
              << (estimated_size / (1024.0 * 1024.0)) << " MB\n";

    CUDA_CHECK(cudaMalloc(&ws_.d_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_cb1_sumq, Q_DOCLEN * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&ws_.d_emb_ids, ws_.max_stage1_pairs * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_offsets, (Q_DOCLEN + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_emb_dists, ws_.max_stage1_pairs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_doc_ids, ws_.max_stage1_pairs * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&ws_.d_sorted_doc_ids, ws_.max_stage1_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_unique_doc_ids, doc_buf_rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_stage1_doc_scores, doc_buf_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_doc_query_max, doc_buf_rows * Q_DOCLEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_num_unique_docs, sizeof(int)));
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
    CUDA_CHECK(cudaMalloc(&ws_.d_ht_keys, ws_.ht_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_ht_vals, ws_.ht_capacity * sizeof(int)));
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

#ifndef GPU_MVR_OVERLAP_STAGE23
    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, temp_bytes,
        ws_.d_doc_scores, ws_.d_topk_scores,
        ws_.d_selected_indices, ws_.d_topk_indices,
        (int)ws_.max_stage2_candidates, 0, 32
    );
    ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);
#endif

    temp_bytes = 0;
    cub::DeviceScan::InclusiveScan(
        nullptr, temp_bytes,
        ws_.d_pair_offsets + 1, ws_.d_pair_offsets + 1,
        thrust::plus<size_t>(), (int)Q_DOCLEN
    );
    ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes);

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

    // CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // std::cout << "GPU memory usage after index upload: "
    //           << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB / "
    //           << (total_mem / (1024.0 * 1024.0)) << " MB\n";

    size_t topk_buf_size = std::max(doc_buf_rows, ws_.max_stage2_candidates);
    CUDA_CHECK(cudaMalloc(&ws_.d_topk_scores, topk_buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_topk_doc_ids, topk_buf_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_topk_indices, topk_buf_size * sizeof(int)));

#ifndef GPU_MVR_OVERLAP_STAGE23
    CUDA_CHECK(cudaMalloc(&ws_.d_candidate_offsets, (ws_.max_stage2_candidates + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&ws_.d_token_dists, ws_.max_stage2_tokens * Q_DOCLEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_doc_scores, ws_.max_stage2_candidates * sizeof(float)));
#endif

    CUDA_CHECK(cudaMalloc(&ws_.d_selected_indices, ws_.max_stage2_k * sizeof(int)));

#ifdef GPU_MVR_USE_LUT
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
#ifndef GPU_MVR_OVERLAP_STAGE23
    CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_batch_scores, ws_.max_stage2_candidates * sizeof(float)));
#endif

#ifdef GPU_MVR_OVERLAP_STAGE23
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

    ws_.max_embs_per_query_bound = nprobe * max_cluster_size;
    CUDA_CHECK(cudaMemset(ws_.d_doc_query_max, 0, doc_buf_rows * Q_DOCLEN * sizeof(float)));
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
    CUDA_CHECK(cudaMemset(ws_.d_ht_keys, 0xff, ws_.ht_capacity * sizeof(int)));  // fill with -1
    CUDA_CHECK(cudaMemset(ws_.d_ht_vals, 0xff, ws_.ht_capacity * sizeof(int)));  // fill with -1
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

// ======================== doc_len ========================

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
        XFER_RECORD_END(ws_.stream_h2d, Q_DOCLEN * PADDED_DIM * sizeof(float) + Q_DOCLEN * sizeof(float), true);
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
    std::vector<size_t> stage2_doc_ids;
    std::vector<float>  stage2_one_bit_dists;

    rank_all_tokens_1bit_gpu(actual_k_stage1, k_rank_all_tokens,
                             stage2_doc_ids, stage2_one_bit_dists,
                             ws_.stream_compute);

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.event_stage2_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventSynchronize(ws_.event_stage2_end));
    }
#endif

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

    auto search_end = std::chrono::high_resolution_clock::now();
    if constexpr (kProfile) {
        float search_wall_ms = std::chrono::duration<float, std::milli>(search_end - search_start).count();
        std::cout << "[SEARCH] Total wall-clock time: " << search_wall_ms << " ms\n";
    }

    return result;
}

// ======================== STAGE 1: GPU ========================

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
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
    size_t doc_matrix_size = ws_.max_compact_docs * Q_DOCLEN;
#else
    size_t doc_matrix_size = ws_.estimated_num_docs * Q_DOCLEN;
#endif
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_memset_start, ws_.stream_d2h));
    }
#endif
    CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_query_max, 0, doc_matrix_size * sizeof(float), ws_.stream_d2h));
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
    CUDA_CHECK(cudaMemsetAsync(ws_.d_ht_keys, 0xff, ws_.ht_capacity * sizeof(int), ws_.stream_d2h));
    CUDA_CHECK(cudaMemsetAsync(ws_.d_ht_vals, 0xff, ws_.ht_capacity * sizeof(int), ws_.stream_d2h));
#else
    CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_touched, 0, ws_.estimated_num_docs * sizeof(int), ws_.stream_d2h));
#endif
    CUDA_CHECK(cudaMemsetAsync(ws_.d_num_unique_docs, 0, sizeof(int), ws_.stream_d2h));
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

    ivf->search_batch_gpu(ws_.d_queries, Q_DOCLEN, nprobe,
                          ws_.d_cagra_dists, ws_.d_cagra_labels, stream);

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_cagra_end, stream));
        CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_start, stream));
    }
#endif

#ifdef GPU_MVR_USE_LUT
    // LUT path: the kernel iterates clusters directly from CAGRA labels,
    // so we skip the expansion kernel and d_emb_ids entirely. This removes
    // the emb_id load → code load dependency chain (400 cycles savings)
    // and the expansion kernel itself (~0.18 ms).
#ifdef GPU_MVR_PROFILE
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

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_end, stream));
    }
#endif
#endif  // GPU_MVR_USE_LUT

    int threads_per_block = 256;

    // The fused stage1 kernel writes directly into d_doc_query_max /
    // d_ht_keys / d_num_unique_docs, so the memset on stream_d2h must
    // complete before we launch it. The memset runs concurrently
    // with CAGRA search, so waiting here is free on the critical path.
    CUDA_CHECK(cudaStreamWaitEvent(stream, ws_.event_h2d_done));

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_start, stream));
    }
#endif
#ifdef GPU_MVR_USE_LUT
    if (use_clustered_) {
        // Clustered layout: flat iteration across all probed clusters with
        // coalesced memory access. Ensure enough blocks for good SM occupancy.
        int blocks_x = std::max(
            (max_cluster_size + threads_per_block - 1) / threads_per_block,
            16);
        dim3 grid(blocks_x, Q_DOCLEN);

        stage1_binary_ip_lut_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_lut, d_clustered_code_, d_clustered_factor_, ws_.d_cb1_sumq,
            ws_.d_cagra_labels, d_cluster_pos_,
            d_clustered_doc_ids_,
            ws_.d_doc_query_max,
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
            ws_.d_ht_keys,
            ws_.d_ht_vals,
            ws_.d_unique_doc_ids,
            ws_.d_num_unique_docs,
            ws_.ht_mask,
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
        throw std::runtime_error("v3 requires clustered gpu_index.bin layout");
    }
#else
    throw std::runtime_error("v3 requires GPU_MVR_USE_LUT");
#endif
    CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_end, stream));
    }
#endif

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_start, stream));
    }
#endif
#ifndef GPU_MVR_USE_LUT
    throw std::runtime_error("v3 requires GPU_MVR_USE_LUT");
#endif
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_end, stream));
    }
#endif

    int h_num_touched = 0;
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
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
    if ((size_t)h_num_touched > ws_.max_compact_docs) {
        std::cerr << "WARNING: h_num_touched (" << h_num_touched
                  << ") exceeds max_compact_docs (" << ws_.max_compact_docs
                  << "). Results may be corrupted. Increase nprobe*max_cluster_size headroom.\n";
    }
#endif
    actual_k_out = std::min((int)k, h_num_touched);

    const int thread_count = SUM_SCORES_WARPS_PER_BLOCK * 32; // 256
    int sparse_blocks = (h_num_touched + SUM_SCORES_WARPS_PER_BLOCK - 1) / SUM_SCORES_WARPS_PER_BLOCK;
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
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
        h_num_touched
#else
        h_num_touched,
        num_docs
#endif
    );
    CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_sum_scores_end, stream));
    }
#endif

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
}

// ======================== rank_stage23_persistent ========================

void gpu_mvr_index::rank_stage23_persistent(
    int num_candidates,
    size_t k,
    size_t k_stage2,
    query_object* queries,
    std::vector<size_t>& result
) {
    rank_stage23_persistent_impl<false>(num_candidates, k, k_stage2, queries, result);
}

template <bool kProfile>
void gpu_mvr_index::rank_stage23_persistent_impl(
    int num_candidates,
    size_t k,
    size_t k_stage2,
    query_object* queries,
    std::vector<size_t>& result
) {
    if (num_candidates == 0) return;

    auto cpu_ms = [](auto a, auto b) { return std::chrono::duration<float, std::milli>(b - a).count(); };

    cudaStream_t stream = ws_.stream_compute;
    cudaStream_t stream_d2h = ws_.stream_d2h;
    int actual_k = (int)std::min(k_stage2, (size_t)num_candidates);

#ifdef GPU_MVR_TIMELINE
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

    int actual_chunks = std::min((int)N_OVERLAP_CHUNKS, num_candidates);
    int cand_chunk_size = (num_candidates + actual_chunks - 1) / actual_chunks;
#ifdef GPU_MVR_TIMELINE
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

#ifdef GPU_MVR_PROFILE
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
#ifdef GPU_MVR_PROFILE
            if constexpr (kProfile) {
                chunk_has_bip[c] = true;
                CUDA_CHECK(cudaEventRecord(chunk_bip_start[c], stream));
            }
#endif
#ifdef GPU_MVR_USE_LUT
            {
                size_t stage2_lut_smem = STAGE2_LUT_SMEM_FLOATS * sizeof(float) + STAGE2_LUT_TILE_Q * sizeof(float);
                stage2_binary_ip_lut_kernel<<<bip_blocks, 256, stage2_lut_smem, stream>>>(
                    ws_.d_lut, d_clustered_code_, d_clustered_factor_,
                    d_token_to_cluster_pos_, ws_.d_cb1_sumq,
                    ws_.d_pst_token_ids + tok_start,
                    ws_.d_token_dists + tok_start,
                    total_tokens, tok_count
                );
            }
#else
            throw std::runtime_error("v3 requires GPU_MVR_USE_LUT");
#endif
#ifdef GPU_MVR_PROFILE
            if constexpr (kProfile) {
                CUDA_CHECK(cudaEventRecord(chunk_bip_end[c], stream));
            }
#endif
            CUDA_CHECK(cudaGetLastError());
        }

#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(chunk_docscore_start[c], stream));
        }
#endif
        doc_score_kernel<<<c_count, score_threads, 0, stream>>>(
            ws_.d_token_dists, ws_.d_pst_candidate_offsets + c_start,
            ws_.d_doc_scores + c_start, total_tokens, c_count
        );
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(chunk_docscore_end[c], stream));
        }
#endif
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(chunk_compute_done[c], stream));
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            if (c == actual_chunks - 1)
                CUDA_CHECK(cudaEventRecord(ws_.s23_pst_kernel_end, stream));
        }
#endif

        int c_start_d2h = c * cand_chunk_size;
        int c_end_d2h = std::min(c_start_d2h + cand_chunk_size, num_candidates);
        int c_count_d2h = c_end_d2h - c_start_d2h;
#ifdef GPU_MVR_TIMELINE
        CUDA_CHECK(cudaEventRecord(tl_c_wait_ev[c], stream_d2h));
#endif
        CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, chunk_compute_done[c], 0));
#ifdef GPU_MVR_TIMELINE
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
#ifdef GPU_MVR_TIMELINE
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
#ifdef GPU_MVR_TIMELINE
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
#ifdef GPU_MVR_TIMELINE
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
            int doc_id = h_candidate_doc_ids[p.second];
            seen_doc_map[doc_id] = true;
            seen_doc_list.push_back(doc_id);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        time_identify_new += std::chrono::duration<double, std::milli>(t3 - t2).count();
#ifdef GPU_MVR_TIMELINE
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
#ifdef GPU_MVR_TIMELINE
        tl_cpu.push_back({tl_cpu_us(t3), tl_cpu_us(t4), "launch_extract_c" + std::to_string(c), "async"});
#endif

        cpu_compute_ip_ex(
            h_candidate_doc_ids, h_sel_indices.data(), h_out_offsets.data(),
            to_refine, ws_.h_pinned_queries, ip_ex_buf.data());
        auto t5 = std::chrono::high_resolution_clock::now();
        time_cpu_ip_ex += std::chrono::duration<double, std::milli>(t5 - t4).count();
#ifdef GPU_MVR_TIMELINE
        tl_cpu.push_back({tl_cpu_us(t4), tl_cpu_us(t5), "cpu_ip_ex_c" + std::to_string(c), "static"});
#endif

        CUDA_CHECK(cudaEventSynchronize(chunk_extract_done[c]));
        auto t6 = std::chrono::high_resolution_clock::now();
        time_wait_extract += std::chrono::duration<double, std::milli>(t6 - t5).count();
#ifdef GPU_MVR_TIMELINE
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
#ifdef GPU_MVR_TIMELINE
        tl_cpu.push_back({tl_cpu_us(t6), tl_cpu_us(t7), "combine_c" + std::to_string(c), "subflow"});
#endif

        total_refined += to_refine;
    }

    auto t_end_phase_c = std::chrono::high_resolution_clock::now();
    double time_phase_c = std::chrono::duration<double, std::milli>(t_end_phase_c - t_start_phase_c).count();
    double total_wall_time = std::chrono::duration<double, std::milli>(t_end_phase_c - t_start_phase_a).count();

#ifdef GPU_MVR_PROFILE
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

#ifdef GPU_MVR_TIMELINE
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamSynchronize(stream_d2h));
        CUDA_CHECK(cudaStreamSynchronize(ws_.stream_extract));

        tl_gpu_compute.push_back({tl_gpu_us(ws_.phase_a_start_event), tl_gpu_us(ws_.phase_a_gather_done_event), "gather_doc_lengths", "async"});
        tl_gpu_compute.push_back({tl_gpu_us(ws_.phase_a_gather_done_event), tl_gpu_us(ws_.phase_a_prefix_done_event), "prefix_sum", "async"});
        tl_gpu_compute.push_back({tl_gpu_us(ws_.phase_a_prefix_done_event), tl_gpu_us(ws_.phase_a_token_ids_done_event), "gather_token_ids", "async"});
        tl_gpu_compute.push_back({tl_gpu_us(ws_.phase_a_token_ids_done_event), tl_gpu_us(ws_.phase_a_d2h_done_event), "d2h_metadata", "async"});

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

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
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
    }
#endif
}

// ======================== STAGE 3: CPU ========================

void gpu_mvr_index::cpu_compute_ip_ex(
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

void gpu_mvr_index::cpu_combine_scores(
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

gpu_mvr_index::~gpu_mvr_index() {
    CUDA_CHECK(cudaDeviceSynchronize());

    // PG_CAGRA stores a RAFT resource whose active stream is rebound to
    // ws_.stream_compute during search. Release it before destroying streams.
    delete ivf;
    ivf = nullptr;

    CUDA_CHECK(cudaFree(d_doc_ptrs_));
    // d_inv_list_ may be allocated in non-LUT path or as fallback in LUT path;
    // cudaFree(nullptr) is a safe no-op
    CUDA_CHECK(cudaFree(d_inv_list_));
    CUDA_CHECK(cudaFree(d_cluster_pos_));

    // Clustered arrays are nullptr when fallback is active; cudaFree(nullptr) is safe
    CUDA_CHECK(cudaFree(d_clustered_code_));
    CUDA_CHECK(cudaFree(d_clustered_factor_));
    CUDA_CHECK(cudaFree(d_clustered_doc_ids_));
    CUDA_CHECK(cudaFree(d_token_to_cluster_pos_));

    CUDA_CHECK(cudaFree(ws_.d_queries));
    CUDA_CHECK(cudaFree(ws_.d_cb1_sumq));
#ifdef GPU_MVR_USE_LUT
    CUDA_CHECK(cudaFree(ws_.d_lut));
#endif
    CUDA_CHECK(cudaFree(ws_.d_emb_ids));
    CUDA_CHECK(cudaFree(ws_.d_pair_offsets));
    CUDA_CHECK(cudaFree(ws_.d_emb_dists));
    CUDA_CHECK(cudaFree(ws_.d_pair_doc_ids));
#ifndef GPU_MVR_OVERLAP_STAGE23
    CUDA_CHECK(cudaFree(ws_.d_candidate_offsets));
    CUDA_CHECK(cudaFree(ws_.d_token_dists));
    CUDA_CHECK(cudaFree(ws_.d_doc_scores));
#endif
    CUDA_CHECK(cudaFree(ws_.d_selected_indices));

    CUDA_CHECK(cudaFree(ws_.d_cagra_dists));
    CUDA_CHECK(cudaFree(ws_.d_cagra_labels));

    CUDA_CHECK(cudaFree(ws_.d_sorted_doc_ids));
    CUDA_CHECK(cudaFree(ws_.d_unique_doc_ids));
    CUDA_CHECK(cudaFree(ws_.d_stage1_doc_scores));
    CUDA_CHECK(cudaFree(ws_.d_doc_query_max));
    CUDA_CHECK(cudaFree(ws_.d_num_unique_docs));
#ifdef GPU_MVR_COMPACT_DOC_BUFFER
    CUDA_CHECK(cudaFree(ws_.d_ht_keys));
    CUDA_CHECK(cudaFree(ws_.d_ht_vals));
#else
    CUDA_CHECK(cudaFree(ws_.d_doc_touched));
#endif
    CUDA_CHECK(cudaFree(ws_.d_cub_temp_storage));
    CUDA_CHECK(cudaFree(ws_.d_topk_scores));
    CUDA_CHECK(cudaFree(ws_.d_topk_doc_ids));
    CUDA_CHECK(cudaFree(ws_.d_topk_indices));

    CUDA_CHECK(cudaFreeHost(ws_.h_pinned_queries));
    CUDA_CHECK(cudaFreeHost(ws_.h_pinned_cb1_sumq));
#ifndef GPU_MVR_OVERLAP_STAGE23
    CUDA_CHECK(cudaFreeHost(ws_.h_pinned_batch_scores));
#endif

#ifdef GPU_MVR_OVERLAP_STAGE23
    CUDA_CHECK(cudaFree(ws_.d_token_dists));
    CUDA_CHECK(cudaFree(ws_.d_pst_candidate_offsets));
    CUDA_CHECK(cudaFree(ws_.d_pst_token_ids));
    CUDA_CHECK(cudaFree(ws_.d_doc_scores));
    CUDA_CHECK(cudaFreeHost(ws_.h_mapped_doc_scores));
    CUDA_CHECK(cudaFreeHost(ws_.h_pinned_pst_candidate_offsets));
    CUDA_CHECK(cudaFreeHost(ws_.h_pinned_pst_candidate_doc_ids));
    CUDA_CHECK(cudaFreeHost(ws_.h_pinned_pst_total_tokens));
    CUDA_CHECK(cudaEventDestroy(ws_.phase_a_start_event));
    CUDA_CHECK(cudaEventDestroy(ws_.phase_a_gather_done_event));
    CUDA_CHECK(cudaEventDestroy(ws_.phase_a_prefix_done_event));
    CUDA_CHECK(cudaEventDestroy(ws_.phase_a_token_ids_done_event));
    CUDA_CHECK(cudaEventDestroy(ws_.phase_a_d2h_done_event));
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

#ifdef GPU_MVR_PROFILE
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
}
