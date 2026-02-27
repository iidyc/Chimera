// This file is included inside the gpu_mvr_index struct when GPU_MVR_OVERLAP_STAGE23 is defined.
// It provides rank_stage23_persistent() that overlaps GPU Stage 2 COMPUTE with CPU Stage 3.
//
// Architecture: Multi-Chunk Streaming Running Top-K with Compute Overlap
//
//   Phase A: Data prep on GPU (gather lengths, prefix sum, token IDs)
//
//   Phase B: GPU work split into N_OVERLAP_CHUNKS pieces.
//            ALL chunks launched upfront on stream_compute (GPU never waits for CPU).
//            Per-chunk compute events for gating D2H.
//
//   Phase C: Streaming CPU processing (overlapped with GPU):
//     For each chunk c = 0..N-1:
//       1. D2H doc_scores for chunk c (gated on chunk_compute_done[c])
//       2. CPU selects running top-k from all scores seen so far [0, c_end)
//       3. Identifies NEW docs in running top-k not yet refined
//       4. GPU extract: compacts selected docs' 1-bit dists (already computed in
//          Stage 2 binary_ip) into contiguous buffer via extract_one_bit_dists_kernel,
//          then D2H to pinned host memory. Runs async on stream_d2h.
//       5. CPU computes ip_ex_bits from CPU ex_code_ (OVERLAPPED with GPU extract).
//       6. After GPU extract D2H completes, CPU combines 1-bit dists + ip_ex_bits
//          for final refined scores → heap.
//     Running top-k evolves as more scores arrive — later chunks may displace
//     earlier candidates, and only newly-appeared docs get refined.
//
//   Phase D: Extract final top-k from heap (trivial).
//
// Key benefits:
//   - CPU starts refining after 1/N of GPU work (vs waiting for full Stage 2)
//   - Running top-k adapts as GPU reveals more scores → better doc prioritization
//   - Reuses GPU-computed 1-bit distances (no redundant CPU recompute)
//   - GPU extract (< 0.1ms) fully hidden under CPU ip_ex_bits work
//   - CPU ip_ex_bits (~50% of Stage 3 cost) overlaps with GPU chunks

// ============================================================
// Configuration
// ============================================================
static constexpr int N_OVERLAP_CHUNKS = 4;   // Number of GPU scoring chunks
static constexpr int PRELIM_PER_CHUNK = 125;  // Max docs to CPU-refine per chunk

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
    // Phase B: Launch ALL GPU chunk kernels on stream_compute
    //          D2H managed per-chunk in Phase C on stream_d2h
    // ========================================

    int score_threads = 128;
    while (score_threads < Q_DOCLEN && score_threads < 256) score_threads *= 2;
    score_threads = std::min(score_threads, 256);

    // Chunk boundaries (candidates are sorted by Stage 1 score descending)
    int actual_chunks = std::min((int)N_OVERLAP_CHUNKS, num_candidates);
    int cand_chunk_size = (num_candidates + actual_chunks - 1) / actual_chunks;

    // Create per-chunk compute events
    cudaEvent_t chunk_compute_done[N_OVERLAP_CHUNKS];
    for (int c = 0; c < actual_chunks; c++)
        CUDA_CHECK(cudaEventCreateWithFlags(&chunk_compute_done[c], cudaEventDisableTiming));

    // Launch all chunks: binary_ip + doc_score → record compute event
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

        // Record compute completion (gates D2H in Phase C)
        CUDA_CHECK(cudaEventRecord(chunk_compute_done[c], stream));
    }

    // Record event after ALL GPU scoring (for timing measurement)
    CUDA_CHECK(cudaEventRecord(ws_.pst_extract_done, stream));

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

    // Pre-allocate reusable buffers
    std::vector<int> running_indices;
    running_indices.reserve(num_candidates);
    std::vector<int> h_sel_indices(PRELIM_PER_CHUNK);
    std::vector<size_t> h_out_offsets(PRELIM_PER_CHUNK + 1);
    // ip_ex_buf: intermediate storage for CPU-computed ex-bits inner products
    // Indexed as [flat_token_offset * Q_DOCLEN + query_idx]
    std::vector<float> ip_ex_buf((size_t)PRELIM_PER_CHUNK * max_doc_len * Q_DOCLEN);
    std::vector<std::pair<float, int>> refined_scores(PRELIM_PER_CHUNK);

    for (int c = 0; c < actual_chunks; c++) {
        // --- D2H doc_scores for this chunk ---
        auto t0 = std::chrono::high_resolution_clock::now();
        int c_start = c * cand_chunk_size;
        int c_end = std::min(c_start + cand_chunk_size, num_candidates);
        int c_count = c_end - c_start;
        CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, chunk_compute_done[c], 0));
        CUDA_CHECK(cudaMemcpyAsync(
            ws_.h_mapped_doc_scores + c_start,
            ws_.d_doc_scores + c_start,
            c_count * sizeof(float),
            cudaMemcpyDeviceToHost, stream_d2h));
        CUDA_CHECK(cudaStreamSynchronize(stream_d2h));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_wait_d2h += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // --- CPU: running top-k from all scores seen so far [0, c_end) ---
        running_indices.resize(c_end);
        std::iota(running_indices.begin(), running_indices.end(), 0);
        int run_k = std::min(actual_k, c_end);
        std::nth_element(running_indices.begin(), running_indices.begin() + run_k,
                         running_indices.end(),
                         [this](int a, int b) {
                             return ws_.h_mapped_doc_scores[a] > ws_.h_mapped_doc_scores[b];
                         });
        auto t2 = std::chrono::high_resolution_clock::now();
        time_running_topk += std::chrono::duration<double, std::milli>(t2 - t1).count();

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

        if (new_docs.empty()) continue;
        std::sort(new_docs.begin(), new_docs.end(), std::greater<>());
        int to_refine = std::min((int)PRELIM_PER_CHUNK, (int)new_docs.size());
        new_docs.resize(to_refine);
        for (const auto& p : new_docs) {
            seen_doc_ids.insert(h_candidate_doc_ids[p.second]);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        time_identify_new += std::chrono::duration<double, std::milli>(t3 - t2).count();

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
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_selected_indices, h_sel_indices.data(),
                                    to_refine * sizeof(int),
                                    cudaMemcpyHostToDevice, stream_d2h));
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_out_offsets, h_out_offsets.data(),
                                    (to_refine + 1) * sizeof(size_t),
                                    cudaMemcpyHostToDevice, stream_d2h));
        // Extract kernel: compacts scattered 1-bit dists into contiguous output
        extract_one_bit_dists_kernel<<<to_refine, 256, 0, stream_d2h>>>(
            ws_.d_token_dists, ws_.d_pst_candidate_offsets,
            ws_.d_selected_indices, ws_.d_out_one_bit_dists,
            ws_.d_out_offsets, total_tokens, (size_t)to_refine
        );
        CUDA_CHECK(cudaGetLastError());
        // D2H extracted dists (contiguous, async — pinned target)
        CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_dists, ws_.d_out_one_bit_dists,
                                    total_sel_tokens * Q_DOCLEN * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream_d2h));
        auto t4 = std::chrono::high_resolution_clock::now();
        time_gpu_extract += std::chrono::duration<double, std::milli>(t4 - t3).count();

        // --- CPU: compute ip_ex_bits (OVERLAPPED with GPU extract + D2H) ---
        // This is the only CPU-intensive part of Stage 3 refinement.
        // It runs in parallel with the GPU extract kernel on stream_d2h.
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < to_refine; i++) {
            int doc_id = h_candidate_doc_ids[h_sel_indices[i]];
            size_t doc_start = doc_ptrs_[doc_id];
            size_t n_tok = doc_ptrs_[doc_id + 1] - doc_start;
            for (size_t j = 0; j < Q_DOCLEN; j++) {
                for (size_t t = 0; t < n_tok; t++) {
                    size_t tid = doc_start + t;
                    ip_ex_buf[(h_out_offsets[i] + t) * Q_DOCLEN + j] = ip_ex_bits(
                        queries + j,
                        &ex_code_[tid * PADDED_DIM * ex_bits / 8],
                        ip_func_,
                        PADDED_DIM
                    );
                }
            }
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        time_cpu_ip_ex += std::chrono::duration<double, std::milli>(t5 - t4).count();

        // --- Wait for GPU extract D2H to complete ---
        CUDA_CHECK(cudaStreamSynchronize(stream_d2h));
        auto t6 = std::chrono::high_resolution_clock::now();
        time_wait_extract += std::chrono::duration<double, std::milli>(t6 - t5).count();

        // --- Combine: GPU 1-bit dists + CPU ip_ex_bits → final scores ---
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < to_refine; i++) {
            int doc_id = h_candidate_doc_ids[h_sel_indices[i]];
            size_t doc_start = doc_ptrs_[doc_id];
            size_t n_tok = doc_ptrs_[doc_id + 1] - doc_start;
            float doc_score = 0.0f;
            for (size_t j = 0; j < Q_DOCLEN; j++) {
                float max_ts = -std::numeric_limits<float>::infinity();
                for (size_t t = 0; t < n_tok; t++) {
                    size_t tid = doc_start + t;
                    float combined = combine_dists(
                        queries + j,
                        ws_.h_pinned_dists[(h_out_offsets[i] + t) * Q_DOCLEN + j],
                        ip_ex_buf[(h_out_offsets[i] + t) * Q_DOCLEN + j],
                        one_bit_factor_[tid],
                        ex_factor_[tid],
                        ex_bits
                    );
                    max_ts = std::max(max_ts, combined);
                }
                doc_score += max_ts;
            }
            refined_scores[i] = {doc_score, doc_id};
        }

        // Add refined results to heap
        for (int i = 0; i < to_refine; i++) {
            cpu_heap.emplace(refined_scores[i].first, (size_t)refined_scores[i].second);
        }
        auto t7 = std::chrono::high_resolution_clock::now();
        time_combine += std::chrono::duration<double, std::milli>(t7 - t6).count();

        total_refined += to_refine;
    }

    auto t_end_phase_c = std::chrono::high_resolution_clock::now();
    double time_phase_c = std::chrono::duration<double, std::milli>(t_end_phase_c - t_start_phase_c).count();

    printf("[PROFILE] Phase C: wait_d2h=%.3f ms, topk=%.3f ms, identify=%.3f ms, "
           "gpu_extract=%.3f ms, cpu_ip_ex=%.3f ms, wait_extract=%.3f ms, "
           "combine=%.3f ms (total=%.3f ms, %d docs)\n",
           time_wait_d2h, time_running_topk, time_identify_new,
           time_gpu_extract, time_cpu_ip_ex, time_wait_extract,
           time_combine, time_phase_c, total_refined);

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
