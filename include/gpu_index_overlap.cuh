// This file is included inside the gpu_mvr_index struct when GPU_MVR_OVERLAP_STAGE23 is defined.
// It provides rank_stage23_persistent() that overlaps GPU Stage 2 with CPU Stage 3.
//
// Pipeline:
//   Phase A: Data prep on GPU (gather lengths, prefix sum, token IDs)
//   Phase B: GPU binary IP + doc_score for ALL candidates (no per-chunk sync overhead)
//            + async D2H of doc scores
//   Phase C: CPU sorts scores, selects top-k_stage2
//   Phase D: GPU extract top-k dists in chunks → D2H each chunk
//            CPU refines each chunk as it arrives (overlapped with GPU extract)
//
// Key improvements:
//   - No GPU-side heap kernel or extract/transpose for ALL candidates
//   - No per-chunk cudaStreamSynchronize during binary IP (saves ~0.5ms)
//   - Only refine top-k_stage2 candidates (300 not 5000)
//   - GPU extract + D2H overlaps with CPU refinement

// ============================================================
// rank_stage23_persistent: fused Stage 2 + Stage 3 with overlap
// ============================================================
inline void rank_stage23_persistent(
    int num_candidates,
    size_t k,
    size_t k_stage2,        // k_rank_all_tokens (e.g., 300)
    query_object* queries,
    std::vector<size_t>& result
) {
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
    CUDA_CHECK(cudaMemcpyAsync(&total_tokens, ws_.d_pst_candidate_offsets + num_candidates,
                                sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws_.v_pst_candidate_offsets.data(), ws_.d_pst_candidate_offsets,
                                (num_candidates + 1) * sizeof(size_t),
                                cudaMemcpyDeviceToHost, stream));

    std::vector<int> h_candidate_doc_ids(num_candidates);
    CUDA_CHECK(cudaMemcpyAsync(h_candidate_doc_ids.data(), ws_.d_topk_doc_ids,
                                num_candidates * sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ========================================
    // Phase B: GPU binary IP + doc scores → D2H scores
    // ========================================

#ifdef GPU_MVR_PROFILE
    ws_.s23_heap_total_us = 0;
    CUDA_CHECK(cudaEventRecord(ws_.s23_pst_kernel_start, stream));
    auto cpu_start = std::chrono::high_resolution_clock::now();
#endif

    // Score threads for doc_score_kernel
    int score_threads = 128;
    while (score_threads < Q_DOCLEN && score_threads < 256) score_threads *= 2;
    score_threads = std::min(score_threads, 256);

    // Binary IP for all tokens (single launch, maximum GPU utilization)
    if (total_tokens > 0) {
        int bip_blocks = (total_tokens + 255) / 256;
        stage2_binary_ip_kernel_v2<<<bip_blocks, 256, 0, stream>>>(
            ws_.d_queries, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
            ws_.d_pst_token_ids, ws_.d_token_dists,
            total_tokens, total_tokens
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Doc score for all candidates (single launch)
    doc_score_kernel<<<num_candidates, score_threads, 0, stream>>>(
        ws_.d_token_dists, ws_.d_pst_candidate_offsets,
        ws_.d_doc_scores, total_tokens, num_candidates
    );
    CUDA_CHECK(cudaGetLastError());

#ifdef GPU_MVR_PROFILE
    cudaEvent_t true_gpu_end;
    CUDA_CHECK(cudaEventCreate(&true_gpu_end));
    CUDA_CHECK(cudaEventRecord(true_gpu_end, stream));
#endif

    // D2H doc scores (pre-created event, no per-call create/destroy)
    CUDA_CHECK(cudaEventRecord(ws_.pst_compute_done, stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, ws_.pst_compute_done));
    CUDA_CHECK(cudaMemcpyAsync(
        ws_.h_mapped_doc_scores, ws_.d_doc_scores,
        num_candidates * sizeof(float),
        cudaMemcpyDeviceToHost, stream_d2h));

    // Wait for doc scores D2H
    CUDA_CHECK(cudaStreamSynchronize(stream_d2h));

    // ========================================
    // Phase C: CPU selects top-k via nth_element + sort
    // ========================================

    // nth_element is O(n), much faster than partial_sort O(n*log(k)) for large n
    std::vector<int> sorted_indices(num_candidates);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::nth_element(sorted_indices.begin(), sorted_indices.begin() + actual_k,
                     sorted_indices.end(),
                     [this](int a, int b) {
                         return ws_.h_mapped_doc_scores[a] > ws_.h_mapped_doc_scores[b];
                     });
    // Sort top-k for deterministic output (small sort, only 300 elements)
    std::sort(sorted_indices.begin(), sorted_indices.begin() + actual_k,
              [this](int a, int b) {
                  return ws_.h_mapped_doc_scores[a] > ws_.h_mapped_doc_scores[b];
              });

    // Build top-k candidate list and output offsets
    std::vector<int> h_top_k_indices(sorted_indices.begin(), sorted_indices.begin() + actual_k);
    std::vector<size_t> out_offsets(actual_k + 1, 0);
    for (int i = 0; i < actual_k; ++i) {
        int cand_idx = h_top_k_indices[i];
        out_offsets[i + 1] = out_offsets[i] +
            (ws_.v_pst_candidate_offsets[cand_idx + 1] - ws_.v_pst_candidate_offsets[cand_idx]);
    }
    size_t total_selected_tokens = out_offsets[actual_k];

    // ========================================
    // Phase D: GPU extract → chunked D2H → overlapped CPU refine
    // ========================================

    // Upload selection metadata to GPU and run extract
    CUDA_CHECK(cudaMemcpyAsync(ws_.d_selected_indices, h_top_k_indices.data(),
                                actual_k * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws_.d_out_offsets, out_offsets.data(),
                                (actual_k + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream));

    // Single extract kernel for all top-k candidates
    extract_one_bit_dists_kernel<<<actual_k, 256, 0, stream>>>(
        ws_.d_token_dists, ws_.d_pst_candidate_offsets,
        ws_.d_selected_indices, ws_.d_out_one_bit_dists,
        ws_.d_out_offsets, total_tokens, actual_k
    );
    CUDA_CHECK(cudaGetLastError());

    // Record extract done, queue chunked D2H on stream_d2h
    CUDA_CHECK(cudaEventRecord(ws_.pst_extract_done, stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, ws_.pst_extract_done));

    // Compute chunk boundaries (in candidate space)
    constexpr int NUM_D2H_CHUNKS = Workspace::PST_NUM_D2H_CHUNKS;
    int chunk_size = (actual_k + NUM_D2H_CHUNKS - 1) / NUM_D2H_CHUNKS;
    int num_chunks = (actual_k + chunk_size - 1) / chunk_size;

    // Queue ALL D2H chunks with per-chunk events (non-blocking)
    for (int c = 0; c < num_chunks; c++) {
        int c_start = c * chunk_size;
        int c_end = std::min(c_start + chunk_size, actual_k);
        size_t tok_start = out_offsets[c_start];
        size_t tok_end = out_offsets[c_end];
        size_t tok_count = tok_end - tok_start;
        if (tok_count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(
                ws_.h_pinned_dists + tok_start * Q_DOCLEN,
                ws_.d_out_one_bit_dists + tok_start * Q_DOCLEN,
                tok_count * Q_DOCLEN * sizeof(float),
                cudaMemcpyDeviceToHost, stream_d2h));
        }
        CUDA_CHECK(cudaEventRecord(ws_.pst_d2h_chunk_done[c], stream_d2h));
    }

    // Wait ONLY for first chunk D2H (others will be ready by the time threads reach them)
    CUDA_CHECK(cudaEventSynchronize(ws_.pst_d2h_chunk_done[0]));

    // Prefetch ex_code data for top-k while remaining D2H runs
    for (int i = 0; i < actual_k; ++i) {
        int doc_id = h_candidate_doc_ids[h_top_k_indices[i]];
        __builtin_prefetch(&doc_ptrs_[doc_id], 0, 1);
        __builtin_prefetch(&ex_code_[doc_ptrs_[doc_id] * PADDED_DIM * ex_bits / 8], 0, 1);
    }

    // Atomic flags for chunk D2H completion (avoid redundant cudaEventSynchronize calls)
    std::atomic<int> chunk_synced[NUM_D2H_CHUNKS];
    chunk_synced[0].store(1, std::memory_order_relaxed); // First chunk already synced
    for (int c = 1; c < num_chunks; c++)
        chunk_synced[c].store(0, std::memory_order_relaxed);

#ifdef GPU_MVR_PROFILE
    auto heap_start = std::chrono::high_resolution_clock::now();
#endif

    // Barrier-free overlapped CPU refine: single OMP region with dynamic scheduling
    // Threads naturally process chunk 0 first (indices 0..chunk_size-1).
    // By the time any thread reaches chunk 1+, those D2H chunks are already complete.
    std::priority_queue<std::pair<float, size_t>> cpu_heap;
    int total_refined = actual_k;

    #pragma omp parallel
    {
        std::priority_queue<std::pair<float, size_t>> local_heap;

        #pragma omp for schedule(dynamic, 4)
        for (int i = 0; i < actual_k; i++) {
            // Check if this candidate's D2H chunk is ready
            int c = i / chunk_size;
            if (c > 0 && !chunk_synced[c].load(std::memory_order_acquire)) {
                cudaEventSynchronize(ws_.pst_d2h_chunk_done[c]);
                chunk_synced[c].store(1, std::memory_order_release);
            }

            int cand_idx = h_top_k_indices[i];
            int doc_id = h_candidate_doc_ids[cand_idx];
            size_t tok_offset = out_offsets[i];
            size_t num_tokens_doc = out_offsets[i + 1] - tok_offset;

            float doc_score = 0.0f;
            for (size_t j = 0; j < Q_DOCLEN; ++j) {
                float max_token_score = -std::numeric_limits<float>::infinity();
                for (size_t t = 0; t < num_tokens_doc; ++t) {
                    size_t global_tid = doc_ptrs_[doc_id] + t;
                    float one_bit_dist = ws_.h_pinned_dists[(tok_offset + t) * Q_DOCLEN + j];
                    float dist = distance_ex_bits(
                        queries + j,
                        &ex_code_[global_tid * PADDED_DIM * ex_bits / 8],
                        ex_bits,
                        ip_func_,
                        one_bit_dist,
                        one_bit_factor_[global_tid],
                        ex_factor_[global_tid],
                        PADDED_DIM
                    );
                    max_token_score = std::max(max_token_score, dist);
                }
                doc_score += max_token_score;
            }
            local_heap.emplace(doc_score, (size_t)doc_id);
        }

        #pragma omp critical
        {
            while (!local_heap.empty()) {
                cpu_heap.push(local_heap.top());
                local_heap.pop();
            }
        }
    }

#ifdef GPU_MVR_PROFILE
    auto heap_end = std::chrono::high_resolution_clock::now();
    ws_.s23_heap_total_us = std::chrono::duration<float, std::micro>(heap_end - heap_start).count();
#endif

    // ========================================
    // Extract final top-k results
    // ========================================
    result.clear();
    for (size_t i = 0; i < k && !cpu_heap.empty(); ++i) {
        result.push_back(cpu_heap.top().second);
        cpu_heap.pop();
    }

#ifdef GPU_MVR_PROFILE
    CUDA_CHECK(cudaEventRecord(ws_.s23_pst_kernel_end, stream));
    CUDA_CHECK(cudaEventSynchronize(ws_.s23_pst_kernel_end));
    float kernel_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ws_.s23_pst_kernel_start, ws_.s23_pst_kernel_end));
    ws_.s23_pst_kernel_ms = kernel_ms;

    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_total_us = std::chrono::duration<float, std::micro>(cpu_end - cpu_start).count();
    ws_.s23_cpu_total_us = cpu_total_us;

    std::cout << "[PROFILE] Persistent Stage 2+3:\n";
    float true_gpu_ms;
    CUDA_CHECK(cudaEventElapsedTime(&true_gpu_ms, ws_.s23_pst_kernel_start, true_gpu_end));
    CUDA_CHECK(cudaEventDestroy(true_gpu_end));
    std::cout << "[PROFILE]   True GPU kernel time      : " << true_gpu_ms << " ms\n";
    std::cout << "[PROFILE]   GPU kernel time           : " << kernel_ms << " ms\n";
    std::cout << "[PROFILE]   Total wall time (CPU)     : " << (cpu_total_us / 1000.0f) << " ms\n";
    std::cout << "[PROFILE]   CPU refinement time       : " << (ws_.s23_heap_total_us / 1000.0f) << " ms\n";
    std::cout << "[PROFILE]   Total candidates refined  : " << total_refined << "\n";
    std::cout << "[PROFILE]   D2H chunks                : " << num_chunks << "\n";
#endif
}
