#pragma once

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cfloat>
#include <limits>
#include <random>
#include <cstring>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <immintrin.h>

#define EIGEN_NO_CUDA

#include "io.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "quantization.hpp"
#include "estimator.hpp"
#include "query.hpp"
#include "ivf_pg.hpp"
#include "gpu_config.cuh"


#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

// ---------------------------------------------------------------------------
// GPU-accelerated index build pipeline
// ---------------------------------------------------------------------------
// 1. Randomly sample centroids + GPU brute-force nearest-centroid assignment
// 2. Build cuvs CAGRA graph on the centroids
// 3. Quantize data (1-bit + extra-bits via RaBitQ)
// 4. Assemble IVF_PG and serialize everything to disk
// ---------------------------------------------------------------------------
inline void build_index(
    const float* data,
    size_t n,
    size_t d,
    size_t n_clusters,
    size_t ex_bits,
    const std::vector<int>& doc_lens,
    const std::string& filename)
{
    // ------------------------------------------------------------------
    // Step 1: Randomly sample centroids and assign embeddings
    // ------------------------------------------------------------------
    std::cout << "[build_index] Step 1: Randomly sampling " << n_clusters
              << " centroids from " << n << " vectors (d=" << d << ") ..." << std::endl;

    // Randomly sample n_clusters indices without replacement
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(n_clusters);

    // Copy sampled centroids into a contiguous buffer
    std::vector<float> centroids(n_clusters * d);
    for (size_t i = 0; i < n_clusters; ++i) {
        std::memcpy(&centroids[i * d], &data[indices[i] * d], d * sizeof(float));
    }

    // GPU brute-force nearest-centroid assignment using cuvs
    raft::resources res;
    auto cuda_stream = raft::resource::get_cuda_stream(res);

    float* d_centroids_ptr;
    cudaMalloc(&d_centroids_ptr, n_clusters * d * sizeof(float));
    cudaMemcpy(d_centroids_ptr, centroids.data(), n_clusters * d * sizeof(float), cudaMemcpyHostToDevice);

    auto bf_index = cuvs::neighbors::brute_force::build(
        res,
        raft::make_device_matrix_view<const float>(d_centroids_ptr, (int64_t)n_clusters, (int64_t)d),
        cuvs::distance::DistanceType::InnerProduct
    );

    // Assign each embedding to its nearest centroid (batched)
    std::vector<int64_t> list_nos(n);
    constexpr size_t assign_batch = 65536;
    float* d_q_ptr;
    int64_t* d_lab_ptr;
    float* d_dist_ptr;
    cudaMalloc(&d_q_ptr, assign_batch * d * sizeof(float));
    cudaMalloc(&d_lab_ptr, assign_batch * sizeof(int64_t));
    cudaMalloc(&d_dist_ptr, assign_batch * sizeof(float));

    for (size_t start = 0; start < n; start += assign_batch) {
        int64_t cur = std::min(assign_batch, n - start);
        cudaMemcpy(d_q_ptr, data + start * d, cur * d * sizeof(float), cudaMemcpyHostToDevice);

        cuvs::neighbors::brute_force::search(
            res, bf_index,
            raft::make_device_matrix_view<const float>(d_q_ptr, cur, (int64_t)d),
            raft::make_device_matrix_view<int64_t>(d_lab_ptr, cur, (int64_t)1),
            raft::make_device_matrix_view<float>(d_dist_ptr, cur, (int64_t)1)
        );

        cudaMemcpy(list_nos.data() + start, d_lab_ptr, cur * sizeof(int64_t), cudaMemcpyDeviceToHost);
    }
    raft::resource::sync_stream(res);

    cudaFree(d_q_ptr);
    cudaFree(d_lab_ptr);
    cudaFree(d_dist_ptr);
    cudaFree(d_centroids_ptr);

    std::cout << "[build_index] Step 1 done. Centroids sampled, embeddings assigned." << std::endl;

    // Load rotator early — needed for centroid rotation (Step 2) and quantization (Step 4)
    Rotator<float>* rotator = choose_rotator<float>(d, RotatorType::FhtKacRotator, PADDED_DIM);
    std::ifstream rot_in("rotator.bin", std::ios::binary);
    if (!rot_in.is_open()) {
        throw std::runtime_error("Cannot open rotator.bin — please generate it first");
    }
    rotator->load(rot_in);
    rot_in.close();

    // ------------------------------------------------------------------
    // Step 2: Build CAGRA graph on ROTATED centroids
    // ------------------------------------------------------------------
    // At search time, CAGRA is queried with rotated queries, so the
    // centroids must also be in the rotated space for correct retrieval.
    std::cout << "[build_index] Step 2: Rotating centroids and building CAGRA graph on "
              << n_clusters << " centroids ..." << std::endl;

    std::vector<float> rotated_centroids(n_clusters * PADDED_DIM);
    for (size_t i = 0; i < n_clusters; ++i) {
        rotator->rotate(&centroids[i * d], &rotated_centroids[i * PADDED_DIM]);
    }

    PG_CAGRA* pg_cagra = new PG_CAGRA(n_clusters, PADDED_DIM);
    pg_cagra->build_index(rotated_centroids.data());

    std::cout << "[build_index] Step 2 done. CAGRA graph built." << std::endl;

    // ------------------------------------------------------------------
    // Step 3: Assemble IVF_PG
    // ------------------------------------------------------------------
    std::cout << "[build_index] Step 3: Assembling IVF_PG ..." << std::endl;

    IVF_PG* ivf = new IVF_PG(n_clusters, d, PGType::CAGRA);
    // Replace the default-constructed PG_CAGRA with the one we already built
    delete ivf->pg_index;
    ivf->pg_index = pg_cagra;
    // Populate inv_list and cluster_pos from assignments
    ivf->build_from_assignments(list_nos.data(), n);
    ivf->save(filename);

    std::cout << "[build_index] Step 3 done. IVF_PG assembled and saved." << std::endl;

    // ------------------------------------------------------------------
    // Step 4: Quantize data (1-bit + extra-bits)
    // ------------------------------------------------------------------
    std::cout << "[build_index] Step 4: Quantizing " << n << " vectors ..." << std::endl;

    std::vector<char>  one_bit_code(n * PADDED_DIM / 8);
    std::vector<char>  ex_code(n * PADDED_DIM * ex_bits / 8);
    std::vector<float> one_bit_factor(n);
    std::vector<float> ex_factor(n);

    size_t batch_size = 10240;
    for (size_t start = 0; start < n; start += batch_size) {
        size_t end = std::min(start + batch_size, n);
        size_t cur_batch = end - start;

        std::vector<float> rotated(cur_batch * PADDED_DIM);
#pragma omp parallel for
        for (size_t i = 0; i < cur_batch; ++i) {
            rotator->rotate(&data[(start + i) * d], &rotated[i * PADDED_DIM]);

            encode_one_bit(
                &rotated[i * PADDED_DIM],
                PADDED_DIM,
                reinterpret_cast<uint64_t*>(&one_bit_code[(start + i) * PADDED_DIM / 8]),
                &one_bit_factor[start + i]);

            encode_ex_bits(
                &rotated[i * PADDED_DIM],
                PADDED_DIM,
                ex_bits,
                reinterpret_cast<uint8_t*>(&ex_code[(start + i) * PADDED_DIM * ex_bits / 8]),
                &ex_factor[start + i]);
        }
    }

    std::cout << "[build_index] Step 4 done. Quantization complete." << std::endl;

    // ------------------------------------------------------------------
    // Step 5: Serialize index (matches gpu_mvr_index constructor format)
    // ------------------------------------------------------------------
    std::cout << "[build_index] Step 5: Saving index to " << filename << " ..." << std::endl;

    {
        std::ofstream of(filename, std::ios::binary);
        size_t padded_dim = PADDED_DIM;
        of.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
        of.write(reinterpret_cast<const char*>(&d), sizeof(size_t));
        of.write(reinterpret_cast<const char*>(&n_clusters), sizeof(size_t));
        of.write(reinterpret_cast<const char*>(&ex_bits), sizeof(size_t));
        of.write(reinterpret_cast<const char*>(&padded_dim), sizeof(size_t));
        of.write(one_bit_code.data(), one_bit_code.size());
        of.write(ex_code.data(), ex_code.size());
        of.write(reinterpret_cast<const char*>(one_bit_factor.data()), n * sizeof(float));
        of.write(reinterpret_cast<const char*>(ex_factor.data()), n * sizeof(float));
        of.close();
    }

    delete ivf;
    delete rotator;

    std::cout << "[build_index] Done. Index saved to: " << filename << std::endl;
}


// ---------------------------------------------------------------------------
// Patch utility: rebuild only the CAGRA graph with rotated centroids
// ---------------------------------------------------------------------------
// Recomputes cluster centroids from raw data + IVF assignments, rotates them,
// builds a new CAGRA graph, and overwrites only the .ivf.cagra file.
// Much faster than a full rebuild since IVF assignments and quantization are kept.
// ---------------------------------------------------------------------------
inline void rebuild_cagra_rotated(
    size_t n_clusters,
    size_t d,
    const std::string& filename)
{
    // 1. Load existing centroids from the saved CAGRA index (cuvs format)
    std::cout << "[rebuild_cagra] Loading CAGRA from " << filename << ".ivf.cagra ..." << std::endl;
    PG_CAGRA old_pg(n_clusters, d);
    old_pg.load(filename + ".ivf");
    // Extract dataset from GPU-resident CAGRA index
    auto dataset = old_pg.index_cagra->dataset();
    std::vector<float> raw_centroids(n_clusters * d);
    cudaMemcpy(raw_centroids.data(), dataset.data_handle(),
               n_clusters * d * sizeof(float), cudaMemcpyDeviceToHost);

    // // 2. Load rotator and rotate centroids
    Rotator<float>* rotator = choose_rotator<float>(d, RotatorType::FhtKacRotator, PADDED_DIM);
    std::ifstream rot_in("rotator.bin", std::ios::binary);
    if (!rot_in.is_open()) {
        throw std::runtime_error("Cannot open rotator.bin");
    }
    rotator->load(rot_in);
    rot_in.close();

    // std::cout << "[rebuild_cagra] Rotating " << n_clusters << " centroids ..." << std::endl;
    // std::vector<float> rotated_centroids(n_clusters * PADDED_DIM);
    // #pragma omp parallel for
    // for (size_t i = 0; i < n_clusters; ++i) {
    //     rotator->rotate(&raw_centroids[i * d], &rotated_centroids[i * PADDED_DIM]);
    // }

    std::ofstream centroids_out("rotated_centroids.bin", std::ios::binary);
    centroids_out.write((char*)&n_clusters, sizeof(size_t));
    centroids_out.write((char*)&d, sizeof(size_t));
    centroids_out.write((char*)raw_centroids.data(), raw_centroids.size() * sizeof(float));
    centroids_out.close();
    std::cout << "[rebuild_cagra] Rotated centroids saved to rotated_centroids.bin for inspection." << std::endl;

    size_t num_q, q_doclen_file;
    std::vector<float> Q = load_query(q_doclen_file, num_q, d);
    std::vector<float> rotated_Q(num_q * PADDED_DIM);
#pragma omp parallel for
    for (size_t i = 0; i < num_q; ++i) {
        rotator->rotate(&Q[i * d], &rotated_Q[i * PADDED_DIM]);
    }
    std::ofstream query_out("rotated_queries.bin", std::ios::binary);
    query_out.write((char*)&num_q, sizeof(size_t));
    query_out.write((char*)&d, sizeof(size_t));
    query_out.write((char*)rotated_Q.data(), rotated_Q.size() * sizeof(float));
    query_out.close();
    std::cout << "[rebuild_cagra] Rotated queries saved to rotated_queries.bin for inspection." << std::endl;

    // // 3. Build new CAGRA graph on rotated centroids
    // std::cout << "[rebuild_cagra] Building CAGRA graph ..." << std::endl;
    // PG_CAGRA pg_cagra(n_clusters, PADDED_DIM);
    // pg_cagra.build_index(rotated_centroids.data());

    // // 4. Save only the CAGRA file (overwrite .ivf.cagra)
    // std::cout << "[rebuild_cagra] Saving CAGRA to " << filename << ".ivf.cagra ..." << std::endl;
    // pg_cagra.save(filename + ".ivf");

    // delete rotator;
    // std::cout << "[rebuild_cagra] Done." << std::endl;
}
