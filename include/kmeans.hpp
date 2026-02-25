#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iostream>
#include <unordered_set>
#include <cstring>
#include <omp.h>
#include <immintrin.h>

inline float l2_sqr(const float* a, const float* b, size_t d) {
    // 4 independent accumulators hide FMA latency (~4-5 cycles on modern CPUs).
    // Each accumulator covers a separate 16-float lane, so we advance 64 floats
    // per outer iteration.
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 64 <= d; i += 64) {
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i     ), _mm512_loadu_ps(b + i     ));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
        __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32));
        __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48));
        sum0 = _mm512_fmadd_ps(d0, d0, sum0);
        sum1 = _mm512_fmadd_ps(d1, d1, sum1);
        sum2 = _mm512_fmadd_ps(d2, d2, sum2);
        sum3 = _mm512_fmadd_ps(d3, d3, sum3);
    }
    // Tail: remaining full 16-float blocks
    for (; i + 16 <= d; i += 16) {
        __m512 dv = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        sum0 = _mm512_fmadd_ps(dv, dv, sum0);
    }
    // Reduce the four accumulators
    sum0 = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    float result = _mm512_reduce_add_ps(sum0);
    // Scalar tail for d % 16 != 0
    for (; i < d; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

// Returns the flat index (into the data array) of the nearest vector in doc to query.
inline size_t nearest_in_doc(
    const float* data,
    const int* doc_ptrs,
    size_t doc_id,
    const float* query,
    size_t d
) {
    int start = doc_ptrs[doc_id];
    int end = doc_ptrs[doc_id + 1];
    size_t best_idx = static_cast<size_t>(start);
    float best_dist = std::numeric_limits<float>::min();
    for (int v = start; v < end; ++v) {
        float dist = l2_sqr(data + static_cast<size_t>(v) * d, query, d);
        if (dist > best_dist) {
            best_dist = dist;
            best_idx = static_cast<size_t>(v);
        }
    }
    return best_idx;
}

// Computes dist(S_i, q_k) = min_{p in S_i} ||p - q_k||^2
inline float doc_centroid_dist(
    const float* data,
    const int* doc_ptrs,
    size_t doc_id,
    const float* centroid,
    size_t d
) {
    int start = doc_ptrs[doc_id];
    int end = doc_ptrs[doc_id + 1];
    float best_dist = std::numeric_limits<float>::min();
    for (int v = start; v < end; ++v) {
        float dist = l2_sqr(data + static_cast<size_t>(v) * d, centroid, d);
        best_dist = std::max(best_dist, dist);
    }
    return best_dist;
}

// Sample M distinct indices from [0, total) using O(M) memory.
inline std::vector<size_t> sample_indices(size_t total, size_t M) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<size_t> dist(0, total - 1);
    std::unordered_set<size_t> chosen;
    chosen.reserve(M);
    while (chosen.size() < M) {
        chosen.insert(dist(rng));
    }
    return std::vector<size_t>(chosen.begin(), chosen.end());
}

struct kmeans_result {
    std::vector<int> assignments;   // size N: cluster ID per document
    std::vector<float> centroids;   // size M*d: row-major centroids
};

// Assign every document in [0, N_docs) to its nearest centroid.
inline void assign_docs(
    const float* data,
    const int*   doc_ptrs,
    size_t       N_docs,
    size_t       d,
    size_t       M,
    const std::vector<float>& centroids,
    std::vector<int>&         assignments
) {
#pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < N_docs; ++i) {
        if (doc_ptrs[i] >= doc_ptrs[i + 1]) { assignments[i] = -1; continue; }
        int best_k = 0;
        float best_dist = std::numeric_limits<float>::max();
        for (size_t k = 0; k < M; ++k) {
            float dist = doc_centroid_dist(data, doc_ptrs, i, &centroids[k * d], d);
            if (dist < best_dist) { best_dist = dist; best_k = static_cast<int>(k); }
        }
        assignments[i] = best_k;
    }
}

/**
 * Customized K-means for vector-set data using directed Chamfer-nearest distance.
 *
 * Objective: min_{q_k, z_i} SUM_i min_{p in S_i} ||p - q_{z_i}||^2
 *
 * @param data         All vectors, contiguous row-major (total_vectors * d)
 * @param doc_ptrs     Document boundaries, size N+1. Doc i owns vectors [doc_ptrs[i], doc_ptrs[i+1])
 * @param N            Number of documents (vector sets)
 * @param total_vectors Total number of vectors across all documents
 * @param d            Vector dimension
 * @param M            Number of clusters
 * @param max_iter     Maximum number of Lloyd iterations
 */
inline kmeans_result doc_kmeans(
    const float* data,
    const int* doc_ptrs,
    size_t N,
    size_t total_vectors,
    size_t d,
    size_t M,
    size_t max_iter
) {
    if (M > total_vectors) {
        std::cerr << ">>> Warning: M (" << M << ") > total_vectors (" << total_vectors
                  << "). Clamping M to total_vectors." << std::endl;
        M = total_vectors;
    }

    // --- Initialization: sample M random vectors as initial centroids ---
    std::vector<float> centroids(M * d);
    {
        auto sampled = sample_indices(total_vectors, M);
        for (size_t k = 0; k < M; ++k) {
            std::memcpy(&centroids[k * d], data + sampled[k] * d, d * sizeof(float));
        }
    }

    std::vector<int> assignments(N, 0);

    // Number of power-iteration steps in the update phase
    const size_t n_inner = 2;

    for (size_t iter = 0; iter < max_iter; ++iter) {

        // === Assignment Step (on 10% subset) ===
        assign_docs(data, doc_ptrs, N, d, M, centroids, assignments);

        // === Update Step ===
        // Build cluster membership lists from sampled docs only (serial, O(N_sample))
        std::vector<std::vector<size_t>> cluster_members(M);
        for (size_t doc_id = 0; doc_id < N; ++doc_id) {
            int k = assignments[doc_id];
            if (k >= 0 && static_cast<size_t>(k) < M) {
                cluster_members[k].push_back(doc_id);
            }
        }

        // Update centroids in parallel over clusters.
        // Each cluster's centroid slot is written by exactly one thread -- no synchronization needed.
        size_t num_empty = 0;
        size_t max_cluster_size = 0;

        // cluster_scores[iter1][k]: objective for cluster k after inner iteration iter1
        std::vector<std::vector<float>> cluster_scores(n_inner, std::vector<float>(M, 0.0f));

#pragma omp parallel for schedule(dynamic, 1) reduction(+:num_empty) reduction(max:max_cluster_size)
        for (size_t k = 0; k < M; ++k) {
            size_t count = cluster_members[k].size();
            if (count > max_cluster_size) {
                max_cluster_size = count;
            }
            if (count == 0) {
                // Empty cluster: keep old centroid; cluster_scores[*][k] remains 0
                num_empty += 1;
                continue;
            }
            for (size_t iter1 = 0; iter1 < n_inner; ++iter1) {
                // Power iteration: for each doc in cluster, find its nearest vector to current centroid,
                // then average those vectors to form new centroid. Repeat a few times.
                std::vector<float> temp_centroid(d, 0.0f);
                for (size_t doc_id : cluster_members[k]) {
                    size_t proj_idx = nearest_in_doc(data, doc_ptrs, doc_id, &centroids[k * d], d);
                    for (size_t j = 0; j < d; ++j) {
                        temp_centroid[j] += data[proj_idx * d + j];
                    }
                }
                float inv = 1.0f / static_cast<float>(count);
                for (size_t j = 0; j < d; ++j) {
                    temp_centroid[j] *= inv;
                }
                std::memcpy(&centroids[k * d], temp_centroid.data(), d * sizeof(float));

                // Objective for this cluster after updating centroid
                // Thread k is the sole writer of cluster_scores[*][k] -- no synchronization needed
                float obj = 0.0f;
                for (size_t doc_id : cluster_members[k]) {
                    obj += doc_centroid_dist(data, doc_ptrs, doc_id, &centroids[k * d], d);
                }
                cluster_scores[iter1][k] = obj;
            }
        }

        // Aggregate per-inner-iteration objectives and print
        std::vector<float> total_obj(n_inner, 0.0f);
        for (size_t iter1 = 0; iter1 < n_inner; ++iter1)
            for (size_t k = 0; k < M; ++k)
                total_obj[iter1] += cluster_scores[iter1][k];

        std::cout << ">>> K-means iteration " << (iter + 1) << "/" << max_iter
                  << " | largest cluster: " << max_cluster_size
                  << " | empty clusters: " << num_empty << std::endl;
        for (size_t iter1 = 0; iter1 < n_inner; ++iter1)
            std::cout << "    inner iter " << (iter1 + 1) << "/" << n_inner
                      << " total objective: " << total_obj[iter1] << std::endl;
    }

    // Assign ALL N documents to nearest centroid using converged centroids
    assign_docs(data, doc_ptrs, N, d, M, centroids, assignments);

    return {std::move(assignments), std::move(centroids)};
}
