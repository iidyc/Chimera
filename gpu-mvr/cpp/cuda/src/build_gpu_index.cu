#include "build_gpu_index.hpp"

#define EIGEN_NO_CUDA

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "gpu_config.cuh"
#include "ivf_pg.hpp"
#include "quantization.hpp"

using namespace rabitqlib;

void build_index(
    const float* data,
    size_t n,
    size_t d,
    size_t n_clusters,
    size_t ex_bits,
    const std::vector<int>& /*doc_lens*/,
    const std::string& filename,
    const std::string& bootstrap_centroids,
    const std::string& bootstrap_list_nos)
{
    const bool bootstrap = !bootstrap_centroids.empty() && !bootstrap_list_nos.empty();

    std::vector<float> centroids;
    std::vector<int64_t> list_nos;

    if (bootstrap) {
        std::cout << "[build_index] Bootstrapping from "
                  << bootstrap_centroids << " and " << bootstrap_list_nos
                  << " (skipping steps 1 and 2)" << std::endl;

        {
            std::ifstream cf(bootstrap_centroids, std::ios::binary);
            if (!cf.is_open()) {
                throw std::runtime_error("Cannot open centroids file: " + bootstrap_centroids);
            }
            int32_t fnc = 0, fd = 0;
            cf.read(reinterpret_cast<char*>(&fnc), sizeof(int32_t));
            cf.read(reinterpret_cast<char*>(&fd), sizeof(int32_t));
            if (static_cast<size_t>(fnc) != n_clusters || static_cast<size_t>(fd) != d) {
                throw std::runtime_error(
                    "Centroids file shape mismatch: got (" + std::to_string(fnc) + "," +
                    std::to_string(fd) + "), expected (" + std::to_string(n_clusters) + "," +
                    std::to_string(d) + ")");
            }
            centroids.resize(n_clusters * d);
            cf.read(reinterpret_cast<char*>(centroids.data()),
                    centroids.size() * sizeof(float));
        }

        {
            std::ifstream lf(bootstrap_list_nos, std::ios::binary);
            if (!lf.is_open()) {
                throw std::runtime_error("Cannot open list_nos file: " + bootstrap_list_nos);
            }
            int32_t fn = 0;
            lf.read(reinterpret_cast<char*>(&fn), sizeof(int32_t));
            if (static_cast<size_t>(fn) != n) {
                throw std::runtime_error(
                    "list_nos length mismatch: got " + std::to_string(fn) +
                    ", expected " + std::to_string(n));
            }
            list_nos.resize(n);
            lf.read(reinterpret_cast<char*>(list_nos.data()), n * sizeof(int64_t));
        }

        std::cout << "[build_index] Bootstrap load done." << std::endl;
    } else {
        std::cout << "[build_index] Step 1: Randomly sampling " << n_clusters
                  << " centroids from " << n << " vectors (d=" << d << ") ..." << std::endl;

        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(42);
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(n_clusters);

        centroids.resize(n_clusters * d);
        for (size_t i = 0; i < n_clusters; ++i) {
            std::memcpy(&centroids[i * d], &data[indices[i] * d], d * sizeof(float));
        }

        std::cout << "[build_index] Step 1 done. Sampled centroids." << std::endl;

        std::cout << "[build_index] Step 2: Building temporary non-rotated CAGRA graph"
                  << " for centroid assignment ..." << std::endl;

        PG_CAGRA assignment_cagra(n_clusters, d);
        assignment_cagra.build_index(centroids.data());

        raft::resources res;
        auto cuda_stream = raft::resource::get_cuda_stream(res);

        list_nos.resize(n);
        constexpr size_t assign_batch = 65536;
        float* d_q_ptr;
        uint32_t* d_lab_ptr;
        float* d_dist_ptr;
        cudaMalloc(&d_q_ptr, assign_batch * d * sizeof(float));
        cudaMalloc(&d_lab_ptr, assign_batch * sizeof(uint32_t));
        cudaMalloc(&d_dist_ptr, assign_batch * sizeof(float));
        std::vector<uint32_t> batch_labels(assign_batch);

        for (size_t start = 0; start < n; start += assign_batch) {
            int64_t cur = std::min(assign_batch, n - start);
            cudaMemcpy(d_q_ptr, data + start * d, cur * d * sizeof(float), cudaMemcpyHostToDevice);

            assignment_cagra.search_batch_gpu(d_q_ptr, cur, 1, d_dist_ptr, d_lab_ptr, cuda_stream);

            cudaMemcpy(batch_labels.data(), d_lab_ptr, cur * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            for (int64_t i = 0; i < cur; ++i) {
                list_nos[start + i] = static_cast<int64_t>(batch_labels[i]);
            }
        }
        raft::resource::sync_stream(res);

        cudaFree(d_q_ptr);
        cudaFree(d_lab_ptr);
        cudaFree(d_dist_ptr);

        std::cout << "[build_index] Step 2 done. Embeddings assigned with non-rotated CAGRA." << std::endl;
    }

    Rotator<float>* rotator = choose_rotator<float>(d, RotatorType::FhtKacRotator, PADDED_DIM);
    std::ifstream rot_in("rotator.bin", std::ios::binary);
    if (!rot_in.is_open()) {
        throw std::runtime_error("Cannot open rotator.bin — please generate it first");
    }
    rotator->load(rot_in);
    rot_in.close();

    std::cout << "[build_index] Step 3: Rotating centroids and building persisted CAGRA graph on "
              << n_clusters << " centroids ..." << std::endl;

    std::vector<float> rotated_centroids(n_clusters * PADDED_DIM);
    for (size_t i = 0; i < n_clusters; ++i) {
        rotator->rotate(&centroids[i * d], &rotated_centroids[i * PADDED_DIM]);
    }

    PG_CAGRA* pg_cagra = new PG_CAGRA(n_clusters, PADDED_DIM);
    pg_cagra->build_index(rotated_centroids.data());

    std::cout << "[build_index] Step 3 done. Persisted rotated CAGRA graph built." << std::endl;

    std::cout << "[build_index] Step 4: Assembling IVF_PG ..." << std::endl;

    IVF_PG* ivf = new IVF_PG(n_clusters, d, PGType::CAGRA);
    delete ivf->pg_index;
    ivf->pg_index = pg_cagra;
    ivf->build_from_assignments(list_nos.data(), n);
    ivf->save(filename);

    std::cout << "[build_index] Step 4 done. IVF_PG assembled and saved." << std::endl;

//     std::cout << "[build_index] Step 5: Quantizing " << n << " vectors ..." << std::endl;

//     std::vector<char>  one_bit_code(n * PADDED_DIM / 8);
//     std::vector<char>  full_code(n * PADDED_DIM * (1 + ex_bits) / 8);
//     std::vector<float> one_bit_factor(n);
//     std::vector<float> ex_factor(n);

//     size_t batch_size = 10240;
//     for (size_t start = 0; start < n; start += batch_size) {
//         size_t end = std::min(start + batch_size, n);
//         size_t cur_batch = end - start;

//         std::vector<float> rotated(cur_batch * PADDED_DIM);
// #pragma omp parallel for
//         for (size_t i = 0; i < cur_batch; ++i) {
//             rotator->rotate(&data[(start + i) * d], &rotated[i * PADDED_DIM]);

//             encode_one_bit(
//                 &rotated[i * PADDED_DIM],
//                 PADDED_DIM,
//                 reinterpret_cast<uint64_t*>(&one_bit_code[(start + i) * PADDED_DIM / 8]),
//                 &one_bit_factor[start + i]);

//             encode_full_code(
//                 &rotated[i * PADDED_DIM],
//                 PADDED_DIM,
//                 ex_bits,
//                 reinterpret_cast<uint8_t*>(&full_code[(start + i) * PADDED_DIM * (1 + ex_bits) / 8]),
//                 &ex_factor[start + i]);
//         }
//     }

//     std::cout << "[build_index] Step 5 done. Quantization complete." << std::endl;

//     std::cout << "[build_index] Step 6: Saving index payload to " << filename << " ..." << std::endl;

//     {
//         std::ofstream of(filename, std::ios::binary);
//         size_t padded_dim = PADDED_DIM;
//         of.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
//         of.write(reinterpret_cast<const char*>(&d), sizeof(size_t));
//         of.write(reinterpret_cast<const char*>(&n_clusters), sizeof(size_t));
//         of.write(reinterpret_cast<const char*>(&ex_bits), sizeof(size_t));
//         of.write(reinterpret_cast<const char*>(&padded_dim), sizeof(size_t));
//         of.write(one_bit_code.data(), one_bit_code.size());
//         of.write(full_code.data(), full_code.size());
//         of.write(reinterpret_cast<const char*>(one_bit_factor.data()), n * sizeof(float));
//         of.write(reinterpret_cast<const char*>(ex_factor.data()), n * sizeof(float));
//         of.close();
//     }

    // delete ivf;
    delete rotator;

    std::cout << "[build_index] Done. Index saved to: " << filename << std::endl;
}
