#include "gpu_index_v8.cuh"

#define gpu_mvr_index gpu_mvr_index_v8_base
#define cast_int_size_t cast_int_size_t_v8_base
#include "gpu_index_v3.cu"
#undef cast_int_size_t
#undef gpu_mvr_index

gpu_mvr_index::gpu_mvr_index(
    const std::string& filename,
    const std::vector<int>& doc_lens,
    const gpu_search_runtime_options& runtime_options)
    : gpu_mvr_index_v8_base(filename, doc_lens, runtime_options) {}

std::vector<size_t> gpu_mvr_index::search(const float* queries, size_t k) {
    return search_impl<false>(queries, k);
}

std::vector<size_t> gpu_mvr_index::search_profiled(const float* queries, size_t k) {
    return search_profiled(queries, k, nullptr, true);
}

std::vector<size_t> gpu_mvr_index::search_profiled(
    const float* queries,
    size_t k,
    gpu_search_profile_v8* profile,
    bool print_profile)
{
    return search_impl<true>(queries, k, profile, print_profile);
}

template <bool kProfile>
std::vector<size_t> gpu_mvr_index::search_impl(
    const float* queries,
    size_t k,
    gpu_search_profile_v8* profile,
    bool print_profile)
{
    auto search_start = std::chrono::high_resolution_clock::now();
    gpu_search_profile_v8 local_profile;
    gpu_search_profile_v8* profile_ptr = profile;
    if constexpr (kProfile) {
        if (profile_ptr == nullptr && print_profile) {
            profile_ptr = &local_profile;
        }
        if (profile_ptr != nullptr) {
            *profile_ptr = gpu_search_profile_v8{};
        }
    }
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
#ifdef GPU_MVR_OVERLAP_STAGE23
    gpu_mvr_index_v8_base::rank_stage23_persistent_impl<kProfile>(
        actual_k_stage1,
        k,
        k_rank_all_tokens,
        query_objs.data(),
        result,
        print_profile);
#else
    gpu_mvr_index_v8_base::rank_stage23_persistent_impl<kProfile>(
        actual_k_stage1,
        k,
        k_rank_all_tokens,
        query_objs.data(),
        result,
        print_profile);
#endif

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
            profile_ptr->stage23_time_ms = std::max(0.0f, total_time - stage1_time);
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

    auto search_end = std::chrono::high_resolution_clock::now();
    if constexpr (kProfile) {
        const float search_wall_ms =
            std::chrono::duration<float, std::milli>(search_end - search_start).count();
        if (profile_ptr != nullptr) {
            profile_ptr->search_wall_ms = search_wall_ms;
        }
        if (print_profile) {
            gpu_search_profile_v8 profile_to_print =
                (profile_ptr != nullptr) ? *profile_ptr : gpu_search_profile_v8{};
            profile_to_print.search_wall_ms = search_wall_ms;
            print_gpu_search_profile_v8(profile_to_print, "[PROFILE]");
        }
    }

    return result;
}

void gpu_mvr_index::rank_cluster_dists_gpu(
    query_object* h_query_objs,
    size_t nprobe,
    size_t k,
    int& actual_k_out,
    cudaStream_t stream) {
    rank_cluster_dists_gpu_impl<false>(h_query_objs, nprobe, k, actual_k_out, stream);
}

template <bool kProfile>
void gpu_mvr_index::rank_cluster_dists_gpu_impl(
    query_object* h_query_objs,
    size_t nprobe,
    size_t k,
    int& actual_k_out,
    cudaStream_t stream
) {
    size_t doc_matrix_size = ws_.estimated_num_docs * Q_DOCLEN;
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_memset_start, ws_.stream_d2h));
    }
#endif
    CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_query_max, 0, doc_matrix_size * sizeof(float), ws_.stream_d2h));
    CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_touched, 0, ws_.estimated_num_docs * sizeof(int), ws_.stream_d2h));
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
                          ws_.d_cagra_dists, ws_.d_cagra_labels, stream,
                          static_cast<size_t>(itopk_size));

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_cagra_end, stream));
        CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_start, stream));
    }
#endif

#ifdef GPU_MVR_USE_LUT
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_end, stream));
    }
#endif
    int threads_per_block = 256;
    CUDA_CHECK(cudaStreamWaitEvent(stream, ws_.event_h2d_done));

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_start, stream));
    }
#endif
    if (use_clustered_) {
        int blocks_x = std::max(
            (max_cluster_size + threads_per_block - 1) / threads_per_block,
            16);
        dim3 grid(blocks_x, Q_DOCLEN);

        stage1_binary_ip_lut_v8_align_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_lut, d_clustered_code_, d_clustered_factor_, ws_.d_cb1_sumq,
            ws_.d_cagra_labels, d_cluster_pos_,
            d_clustered_doc_ids_,
            ws_.d_doc_query_max,
            ws_.d_doc_touched,
            ws_.d_unique_doc_ids,
            ws_.d_num_unique_docs,
            num_docs,
            nprobe,
            ivf->n_clusters
        );
    } else {
        throw std::runtime_error("v8 requires clustered cluster_1bit.bin layout");
    }
#else
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

    int total_pairs = 0;
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
        true
    );

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_expansion_end, stream));
    }
#endif

    int threads_per_block = 256;
    CUDA_CHECK(cudaStreamWaitEvent(stream, ws_.event_h2d_done));

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_start, stream));
    }
#endif
    {
        int max_embs_per_query = ws_.max_embs_per_query_bound;
        int blocks_x = (max_embs_per_query + threads_per_block - 1) / threads_per_block;
        dim3 grid(blocks_x, Q_DOCLEN);

        stage1_binary_ip_kernel_v2<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_queries, d_clustered_code_, d_clustered_factor_, ws_.d_cb1_sumq,
            ws_.d_emb_ids, ws_.d_pair_offsets, ws_.d_emb_dists,
            max_embs_per_query
        );
        CUDA_CHECK(cudaGetLastError());

#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_end, stream));
            CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_start, stream));
        }
#endif
        aggregate_stage1_tracked_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_.d_emb_ids,
            ws_.d_emb_dists,
            ws_.d_pair_offsets,
            d_clustered_doc_ids_,
            ws_.d_doc_query_max,
            ws_.d_doc_touched,
            ws_.d_unique_doc_ids,
            ws_.d_num_unique_docs,
            num_docs,
            ws_.max_stage1_pairs,
            max_embs_per_query
        );
        CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
        if constexpr (kProfile) {
            CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_end, stream));
        }
#endif
    }
#endif
#ifdef GPU_MVR_USE_LUT
    CUDA_CHECK(cudaGetLastError());
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_binary_ip_end, stream));
    }
#endif

#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_start, stream));
        CUDA_CHECK(cudaEventRecord(ws_.s1_atomic_agg_end, stream));
    }
#endif
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
    if (static_cast<size_t>(h_num_touched) > ws_.max_stage1_touched_docs) {
        throw std::runtime_error(
            "Stage1 touched-doc count " + std::to_string(h_num_touched) +
            " exceeded workspace bound " +
            std::to_string(ws_.max_stage1_touched_docs));
    }
    actual_k_out = std::min((int)k, h_num_touched);

    const int thread_count = SUM_SCORES_WARPS_PER_BLOCK * 32;
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
        h_num_touched,
        num_docs
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
        CUDA_CHECK(cudaEventRecord(ws_.s1_d2d_start, stream));
    }
#endif
    CUDA_CHECK(cudaMemcpyAsync(ws_.d_topk_doc_ids, ws_.d_sorted_doc_ids,
                               actual_k_out * sizeof(int),
                               cudaMemcpyDeviceToDevice, stream));
#ifdef GPU_MVR_PROFILE
    if constexpr (kProfile) {
        CUDA_CHECK(cudaEventRecord(ws_.s1_d2d_end, stream));
    }
#endif
}
