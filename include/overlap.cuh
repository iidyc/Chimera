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
    // Phase B: Launch ALL GPU chunk kernels + D2H upfront
    //          GPU processes chunks sequentially; D2H streams follow.
    // ========================================

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

    // ========================================
    // Phase C: Streaming CPU processing with running top-k
    //          (overlapped with GPU processing remaining chunks)
    // ========================================

    std::priority_queue<std::pair<float, size_t>> cpu_heap;
    std::unordered_set<int> seen_doc_ids;
    int total_refined = 0;

    std::vector<std::vector<std::pair<float, int>>> new_docs_all(actual_chunks);
    std::vector<std::vector<std::vector<float>>> binary_ip(actual_chunks); // [chunk][new_doc][query * doc_tokens]
    std::vector<std::vector<std::vector<float>>> ex_ip(actual_chunks); // [chunk][new_doc][query * doc_tokens]

    // Batched memory transfer metadata for GPU→CPU d2h of one_bit_dists for prelim refined docs in each chunk
    std::vector<void*> d2h_src(Q_DOCLEN * PRELIM_PER_CHUNK);
    std::vector<void*> d2h_dst(Q_DOCLEN * PRELIM_PER_CHUNK);
    std::vector<size_t> d2h_size(Q_DOCLEN * PRELIM_PER_CHUNK);
    cudaMemcpyAttributes attrs = {};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;

    for (int c = 0; c < actual_chunks; c++) {
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
        std::vector<std::pair<float, int>>& new_docs = new_docs_all[c];
        for (int i = 0; i < run_k; i++) {
            int cand_idx = running_indices[i];
            int doc_id = h_candidate_doc_ids[cand_idx];
            if (seen_doc_ids.find(doc_id) == seen_doc_ids.end()) {
                new_docs.emplace_back(ws_.h_mapped_doc_scores[cand_idx], cand_idx);
            }
        }

        if (new_docs.size() == 0) continue;
        std::sort(new_docs.begin(), new_docs.end(), std::greater<>());
        int to_refine = std::min((int)PRELIM_PER_CHUNK, (int)new_docs.size());
        new_docs.resize(to_refine);
        for (const auto& p : new_docs) {
            seen_doc_ids.insert(h_candidate_doc_ids[p.second]);
        }

        ex_ip[c].resize(new_docs.size());
        // gather binary_ip for new docs (CPU refine uses distance_one_bit, so GPU-independent)
        binary_ip[c].resize(new_docs.size());
        for (size_t i = 0; i < new_docs.size(); ++i) {
            size_t cand_idx = new_docs[i].second;
            size_t doc_tokens = ws_.v_pst_candidate_offsets[cand_idx + 1] - ws_.v_pst_candidate_offsets[cand_idx];
            binary_ip[c][i].resize(Q_DOCLEN * doc_tokens);
            for (size_t j = 0; j < Q_DOCLEN; ++j) {
                d2h_src[i * Q_DOCLEN + j] = (void*)(ws_.d_token_dists + j * total_tokens + ws_.v_pst_candidate_offsets[cand_idx]);
                d2h_dst[i * Q_DOCLEN + j] = (void*)(binary_ip[c][i].data() + j * doc_tokens);
                d2h_size[i * Q_DOCLEN + j] = doc_tokens * sizeof(float);
            }
        }
        CUDA_CHECK(cudaMemcpyBatchAsync(d2h_dst.data(), d2h_src.data(), d2h_size.data(), Q_DOCLEN * new_docs.size(), attrs, nullptr, stream_d2h));

#pragma omp parallel for
        for (size_t i = 0; i < new_docs_all[c].size(); ++i) {
            int cand_idx = new_docs_all[c][i].second;
            int doc_id = h_candidate_doc_ids[cand_idx];
            size_t doc_start = doc_ptrs_[doc_id];
            size_t num_tokens_doc = doc_ptrs_[doc_id + 1] - doc_start;

            for (size_t j = 0; j < Q_DOCLEN; ++j) {
                for (size_t t = 0; t < num_tokens_doc; ++t) {
                    size_t global_tid = doc_start + t;
                    ex_ip[c][i].push_back(ip_ex_bits(
                        queries + j,
                        &ex_code_[global_tid * PADDED_DIM * ex_bits / 8],
                        ip_func_,
                        PADDED_DIM
                    ));
                }
            }
        }

        total_refined += new_docs_all[c].size();
    }

    // ========================================
    // Phase D: Combine binary_ip + ex_ip to get final ranking
    // ========================================

    // Wait for last D2H binary_ip
    CUDA_CHECK(cudaStreamSynchronize(stream_d2h));

    for (int c = 0; c < actual_chunks; c++) {
        for (size_t i = 0; i < new_docs_all[c].size(); ++i) {
            int cand_idx = new_docs_all[c][i].second;
            int doc_id = h_candidate_doc_ids[cand_idx];
            size_t doc_start = doc_ptrs_[doc_id];
            float doc_score = 0.0F;
            for (size_t j = 0; j < Q_DOCLEN; ++j) {
                float max_token_score = -std::numeric_limits<float>::infinity();
                size_t doc_tokens = doc_len(doc_id);
                for (size_t t = 0; t < doc_tokens; ++t) {
                    size_t tid = doc_start + t;
                    float combined_dist = combine_dists(
                        queries + j,
                        binary_ip[c][i][j * doc_tokens + t],
                        ex_ip[c][i][j * doc_tokens + t],
                        one_bit_factor_[tid],
                        ex_factor_[tid],
                        ex_bits
                    );
                    max_token_score = std::max(max_token_score, combined_dist);
                }
                doc_score += max_token_score;
            }
            cpu_heap.emplace(doc_score, doc_id);
        }
    }

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
}
