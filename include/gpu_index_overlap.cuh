// This file is included inside the gpu_mvr_index struct when GPU_MVR_OVERLAP_STAGE23 is defined.
// It provides rank_stage23_persistent() that overlaps GPU Stage 2 COMPUTE with CPU Stage 3.
//
// Architecture: Multi-Chunk Streaming Running Top-K with Compute Overlap
//
//   Phase A: Data prep on GPU (gather lengths, prefix sum, token IDs)
//
//   Phase B: GPU work split into N_OVERLAP_CHUNKS pieces.
//            ALL chunks launched upfront on stream_compute (GPU never waits for CPU).
//            Per-chunk D2H of doc_scores queued on stream_d2h with separate events.
//
//   Phase C: Streaming CPU processing (overlapped with GPU):
//     For each intermediate chunk c = 0..N-2:
//       1. Wait for chunk c D2H (doc scores arrive)
//       2. Select running top-k from ALL scores seen so far [0, c_end)
//       3. Identify NEW docs in running top-k not yet refined
//       4. CPU refines new docs using distance_one_bit on CPU (zero GPU dependency)
//       This overlaps with GPU processing chunks c+1..N-1.
//     Running top-k evolves as more scores arrive — later chunks may displace
//     earlier candidates, and only newly-appeared docs get refined.
//
//   Phase D: After all GPU chunks complete:
//     1. Wait for final chunk D2H
//     2. Select definitive top-k from ALL scores
//     3. GPU extract + D2H one_bit_dists for remaining unrefined docs
//     4. CPU refines remaining docs using GPU-extracted dists
//
// Key benefits:
//   - CPU starts refining after 1/N of GPU work (vs 1/2 in two-chunk approach)
//   - Running top-k adapts as GPU reveals more scores → better doc prioritization
//   - Zero GPU dependency for CPU refine (uses CPU distance_one_bit)
//   - GPU never stalls waiting for CPU

// ============================================================
// Configuration
// ============================================================
static constexpr int N_OVERLAP_CHUNKS = 4;   // Number of GPU scoring chunks
static constexpr int PRELIM_PER_CHUNK = 25;  // Max docs to CPU-refine per chunk

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
    // Phase B: Launch ALL GPU chunk kernels + D2H upfront
    //          GPU processes chunks sequentially; D2H streams follow.
    // ========================================

#ifdef GPU_MVR_PROFILE
    ws_.s23_heap_total_us = 0;
    CUDA_CHECK(cudaEventRecord(ws_.s23_pst_kernel_start, stream));
    auto cpu_start = std::chrono::high_resolution_clock::now();
#endif

    int score_threads = 128;
    while (score_threads < Q_DOCLEN && score_threads < 256) score_threads *= 2;
    score_threads = std::min(score_threads, 256);

    // Chunk boundaries (candidates are sorted by Stage 1 score descending)
    int actual_chunks = std::min((int)N_OVERLAP_CHUNKS, num_candidates);
    int cand_chunk_size = (num_candidates + actual_chunks - 1) / actual_chunks;

    // Create per-chunk compute events (need separate events — can't reuse
    // because stream_d2h waits on them asynchronously and CUDA overwrite semantics
    // would cause race conditions if the same event were re-recorded before
    // stream_d2h consumes the prior recording)
    cudaEvent_t chunk_compute_done[N_OVERLAP_CHUNKS];
    for (int c = 0; c < actual_chunks; c++)
        CUDA_CHECK(cudaEventCreateWithFlags(&chunk_compute_done[c], cudaEventDisableTiming));

    // Launch all chunks: binary_ip + doc_score → event → D2H scores
    for (int c = 0; c < actual_chunks; c++) {
        int c_start = c * cand_chunk_size;
        int c_end = std::min(c_start + cand_chunk_size, num_candidates);
        int c_count = c_end - c_start;
        size_t tok_start = ws_.v_pst_candidate_offsets[c_start];
        size_t tok_end = ws_.v_pst_candidate_offsets[c_end];
        size_t tok_count = tok_end - tok_start;

        // binary_ip for this chunk's tokens
        if (tok_count > 0) {
            int bip_blocks = (tok_count + 255) / 256;
            stage2_binary_ip_kernel_v2<<<bip_blocks, 256, 0, stream>>>(
                ws_.d_queries, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
                ws_.d_pst_token_ids + tok_start,
                ws_.d_token_dists + tok_start,
                total_tokens, tok_count
            );
            CUDA_CHECK(cudaGetLastError());
        }

        // doc_score for this chunk's candidates
        doc_score_kernel<<<c_count, score_threads, 0, stream>>>(
            ws_.d_token_dists, ws_.d_pst_candidate_offsets + c_start,
            ws_.d_doc_scores + c_start, total_tokens, c_count
        );
        CUDA_CHECK(cudaGetLastError());

        // Record compute completion → gate D2H
        CUDA_CHECK(cudaEventRecord(chunk_compute_done[c], stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, chunk_compute_done[c]));
        CUDA_CHECK(cudaMemcpyAsync(
            ws_.h_mapped_doc_scores + c_start,
            ws_.d_doc_scores + c_start,
            c_count * sizeof(float),
            cudaMemcpyDeviceToHost, stream_d2h));
        CUDA_CHECK(cudaEventRecord(ws_.pst_d2h_chunk_done[c], stream_d2h));
    }

    // Record event after ALL GPU scoring (for True GPU time measurement)
    CUDA_CHECK(cudaEventRecord(ws_.pst_extract_done, stream));

#ifdef GPU_MVR_PROFILE
    cudaEvent_t true_gpu_end;
    CUDA_CHECK(cudaEventCreate(&true_gpu_end));
    CUDA_CHECK(cudaEventRecord(true_gpu_end, stream));
    float chunk_refine_us[N_OVERLAP_CHUNKS] = {};
    int chunk_new_docs[N_OVERLAP_CHUNKS] = {};
#endif

    // ========================================
    // Phase C: Streaming CPU processing with running top-k
    //          (overlapped with GPU processing remaining chunks)
    // ========================================

    std::priority_queue<std::pair<float, size_t>> cpu_heap;
    std::unordered_set<int> seen_doc_ids;
    int total_prelim = 0;

    // Process intermediate chunks (not the last — no GPU work left to overlap)
    for (int c = 0; c < actual_chunks - 1; c++) {
        // Wait for chunk c scores D2H (GPU continues with chunks c+1..N-1)
        CUDA_CHECK(cudaEventSynchronize(ws_.pst_d2h_chunk_done[c]));

        int c_end = std::min((c + 1) * cand_chunk_size, num_candidates);

        // Select running top-k from all scores seen so far [0, c_end)
        std::vector<int> running_indices(c_end);
        std::iota(running_indices.begin(), running_indices.end(), 0);
        int run_k = std::min(actual_k, c_end);
        std::nth_element(running_indices.begin(), running_indices.begin() + run_k,
                         running_indices.end(),
                         [this](int a, int b) {
                             return ws_.h_mapped_doc_scores[a] > ws_.h_mapped_doc_scores[b];
                         });

        // Identify NEW docs in running top-k (not yet refined)
        // Sort new docs by score descending → refine highest-scored first
        std::vector<std::pair<float, int>> new_docs;
        for (int i = 0; i < run_k; i++) {
            int cand_idx = running_indices[i];
            int doc_id = h_candidate_doc_ids[cand_idx];
            if (seen_doc_ids.find(doc_id) == seen_doc_ids.end()) {
                new_docs.emplace_back(ws_.h_mapped_doc_scores[cand_idx], cand_idx);
            }
        }
        std::sort(new_docs.begin(), new_docs.end(), std::greater<>());

        // Budget: cap per chunk to fit in the overlap window
        int to_refine = std::min((int)PRELIM_PER_CHUNK, (int)new_docs.size());
        if (to_refine == 0) continue;

#ifdef GPU_MVR_PROFILE
        auto chunk_refine_start = std::chrono::high_resolution_clock::now();
#endif

        // CPU refine new docs using distance_one_bit (ZERO GPU dependency)
        #pragma omp parallel
        {
            std::priority_queue<std::pair<float, size_t>> local_heap;
            std::vector<int> local_seen;

            #pragma omp for schedule(dynamic, 2)
            for (int i = 0; i < to_refine; i++) {
                int cand_idx = new_docs[i].second;
                int doc_id = h_candidate_doc_ids[cand_idx];
                size_t doc_start = doc_ptrs_[doc_id];
                size_t num_tokens_doc = doc_ptrs_[doc_id + 1] - doc_start;

                float doc_score = 0.0f;
                for (size_t j = 0; j < Q_DOCLEN; ++j) {
                    float max_token_score = -std::numeric_limits<float>::infinity();
                    for (size_t t = 0; t < num_tokens_doc; ++t) {
                        size_t global_tid = doc_start + t;
                        float one_bit_dist = distance_one_bit(
                            queries + j,
                            &one_bit_code_[global_tid * CODE_BYTES],
                            one_bit_factor_[global_tid],
                            PADDED_DIM
                        );
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
                local_seen.push_back(doc_id);
            }

            #pragma omp critical
            {
                while (!local_heap.empty()) {
                    cpu_heap.push(local_heap.top());
                    local_heap.pop();
                }
                for (int id : local_seen) seen_doc_ids.insert(id);
            }
        }

        total_prelim += to_refine;

#ifdef GPU_MVR_PROFILE
        auto chunk_refine_end = std::chrono::high_resolution_clock::now();
        chunk_refine_us[c] = std::chrono::duration<float, std::micro>(
            chunk_refine_end - chunk_refine_start).count();
        chunk_new_docs[c] = to_refine;
#endif
    }

#ifdef GPU_MVR_PROFILE
    auto heap_start = std::chrono::high_resolution_clock::now();
    float prelim_refine_us = std::chrono::duration<float, std::micro>(
        heap_start - cpu_start).count();
#endif

    // ========================================
    // Phase D: Final top-k with GPU-extracted dists for remaining docs
    // ========================================

    // Wait for last chunk D2H
    CUDA_CHECK(cudaEventSynchronize(ws_.pst_d2h_chunk_done[actual_chunks - 1]));

    // CPU: select final top-k from ALL candidates' scores
    std::vector<int> final_sorted(num_candidates);
    std::iota(final_sorted.begin(), final_sorted.end(), 0);
    std::nth_element(final_sorted.begin(), final_sorted.begin() + actual_k,
                     final_sorted.end(),
                     [this](int a, int b) {
                         return ws_.h_mapped_doc_scores[a] > ws_.h_mapped_doc_scores[b];
                     });
    // Sort for deterministic output
    std::sort(final_sorted.begin(), final_sorted.begin() + actual_k,
              [this](int a, int b) {
                  return ws_.h_mapped_doc_scores[a] > ws_.h_mapped_doc_scores[b];
              });

    // Identify docs in final top-k not yet refined
    std::vector<int> remaining_top_k;
    for (int i = 0; i < actual_k; ++i) {
        int cand_idx = final_sorted[i];
        int doc_id = h_candidate_doc_ids[cand_idx];
        if (seen_doc_ids.find(doc_id) == seen_doc_ids.end()) {
            remaining_top_k.push_back(i);  // index into final_sorted
        }
    }

    // Build extraction offsets for remaining docs
    int rem_k = (int)remaining_top_k.size();
    std::vector<int> rem_cand_indices(rem_k);
    std::vector<size_t> rem_out_offsets(rem_k + 1, 0);
    for (int i = 0; i < rem_k; ++i) {
        int fi = remaining_top_k[i];
        int cand_idx = final_sorted[fi];
        rem_cand_indices[i] = cand_idx;
        rem_out_offsets[i + 1] = rem_out_offsets[i] +
            (ws_.v_pst_candidate_offsets[cand_idx + 1] - ws_.v_pst_candidate_offsets[cand_idx]);
    }

    // GPU extract + D2H + CPU refine remaining docs (GPU is idle now, fast path)
    if (rem_k > 0) {
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_selected_indices, rem_cand_indices.data(),
                                    rem_k * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_out_offsets, rem_out_offsets.data(),
                                    (rem_k + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream));

        extract_one_bit_dists_kernel<<<rem_k, 256, 0, stream>>>(
            ws_.d_token_dists, ws_.d_pst_candidate_offsets,
            ws_.d_selected_indices, ws_.d_out_one_bit_dists,
            ws_.d_out_offsets, total_tokens, rem_k
        );
        CUDA_CHECK(cudaGetLastError());

        // Chunked D2H → overlapped CPU refine
        constexpr int NUM_D2H_CHUNKS = Workspace::PST_NUM_D2H_CHUNKS;
        int d2h_chunk_size = (rem_k + NUM_D2H_CHUNKS - 1) / NUM_D2H_CHUNKS;
        int num_d2h_chunks = (rem_k + d2h_chunk_size - 1) / d2h_chunk_size;

        CUDA_CHECK(cudaEventRecord(ws_.pst_compute_done, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, ws_.pst_compute_done));

        for (int dc = 0; dc < num_d2h_chunks; dc++) {
            int dc_start = dc * d2h_chunk_size;
            int dc_end = std::min(dc_start + d2h_chunk_size, rem_k);
            size_t tok_start = rem_out_offsets[dc_start];
            size_t tok_end = rem_out_offsets[dc_end];
            size_t tok_count = tok_end - tok_start;
            if (tok_count > 0) {
                CUDA_CHECK(cudaMemcpyAsync(
                    ws_.h_pinned_dists + tok_start * Q_DOCLEN,
                    ws_.d_out_one_bit_dists + tok_start * Q_DOCLEN,
                    tok_count * Q_DOCLEN * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_d2h));
            }
            CUDA_CHECK(cudaEventRecord(ws_.pst_d2h_chunk_done[dc], stream_d2h));
        }

        CUDA_CHECK(cudaEventSynchronize(ws_.pst_d2h_chunk_done[0]));

        // Prefetch
        for (int i = 0; i < rem_k; ++i) {
            int doc_id = h_candidate_doc_ids[rem_cand_indices[i]];
            __builtin_prefetch(&doc_ptrs_[doc_id], 0, 1);
            __builtin_prefetch(&ex_code_[doc_ptrs_[doc_id] * PADDED_DIM * ex_bits / 8], 0, 1);
        }

        std::atomic<int> d2h_synced[NUM_D2H_CHUNKS];
        d2h_synced[0].store(1, std::memory_order_relaxed);
        for (int dc = 1; dc < num_d2h_chunks; dc++)
            d2h_synced[dc].store(0, std::memory_order_relaxed);

        #pragma omp parallel
        {
            std::priority_queue<std::pair<float, size_t>> local_heap;

            #pragma omp for schedule(dynamic, 4)
            for (int i = 0; i < rem_k; i++) {
                int dc = i / d2h_chunk_size;
                if (dc > 0 && !d2h_synced[dc].load(std::memory_order_acquire)) {
                    cudaEventSynchronize(ws_.pst_d2h_chunk_done[dc]);
                    d2h_synced[dc].store(1, std::memory_order_release);
                }

                int cand_idx = rem_cand_indices[i];
                int doc_id = h_candidate_doc_ids[cand_idx];
                size_t tok_offset = rem_out_offsets[i];
                size_t num_tokens_doc = rem_out_offsets[i + 1] - tok_offset;

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
    }

    int total_refined = total_prelim + rem_k;

#ifdef GPU_MVR_PROFILE
    auto heap_end = std::chrono::high_resolution_clock::now();
    ws_.s23_heap_total_us = std::chrono::duration<float, std::micro>(heap_end - cpu_start).count();
#endif

    // ========================================
    // Extract final top-k results
    // ========================================
    result.clear();
    for (size_t i = 0; i < k && !cpu_heap.empty(); ++i) {
        result.push_back(cpu_heap.top().second);
        cpu_heap.pop();
    }

    // Cleanup compute events
    for (int c = 0; c < actual_chunks; c++)
        CUDA_CHECK(cudaEventDestroy(chunk_compute_done[c]));

#ifdef GPU_MVR_PROFILE
    CUDA_CHECK(cudaEventRecord(ws_.s23_pst_kernel_end, stream));
    CUDA_CHECK(cudaEventSynchronize(ws_.s23_pst_kernel_end));
    float kernel_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ws_.s23_pst_kernel_start, ws_.s23_pst_kernel_end));
    ws_.s23_pst_kernel_ms = kernel_ms;

    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_total_us = std::chrono::duration<float, std::micro>(cpu_end - cpu_start).count();
    ws_.s23_cpu_total_us = cpu_total_us;

    std::cout << "[PROFILE] Persistent Stage 2+3 (multi-chunk streaming overlap):\n";
    float true_gpu_ms;
    CUDA_CHECK(cudaEventElapsedTime(&true_gpu_ms, ws_.s23_pst_kernel_start, true_gpu_end));
    CUDA_CHECK(cudaEventDestroy(true_gpu_end));
    std::cout << "[PROFILE]   True GPU kernel time      : " << true_gpu_ms << " ms\n";
    std::cout << "[PROFILE]   GPU kernel time           : " << kernel_ms << " ms\n";
    std::cout << "[PROFILE]   Total wall time (CPU)     : " << (cpu_total_us / 1000.0f) << " ms\n";
    std::cout << "[PROFILE]   Streaming prelim time     : " << (prelim_refine_us / 1000.0f) << " ms\n";
    std::cout << "[PROFILE]   Total CPU refine time     : " << (ws_.s23_heap_total_us / 1000.0f) << " ms\n";
    std::cout << "[PROFILE]   Chunks: " << actual_chunks
              << " (" << cand_chunk_size << " cands/chunk)\n";
    for (int c = 0; c < actual_chunks - 1; c++) {
        std::cout << "[PROFILE]     Chunk " << c
                  << ": " << chunk_new_docs[c] << " new docs, "
                  << (chunk_refine_us[c] / 1000.0f) << " ms refine\n";
    }
    std::cout << "[PROFILE]   Total prelim (CPU 1bit)   : " << total_prelim << " docs\n";
    std::cout << "[PROFILE]   Remaining (GPU extract)   : " << rem_k << " docs\n";
    std::cout << "[PROFILE]   Total candidates refined  : " << total_refined << "\n";
#endif
}
