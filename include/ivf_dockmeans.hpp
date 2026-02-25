#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <cstring>
#include <omp.h>

#include "kmeans.hpp"

struct IVF_DocKMeans {
    size_t n_clusters;
    size_t d;
    std::vector<float> centroids;       // M * d, row-major
    std::vector<int> inv_list;          // doc IDs grouped by cluster (each doc appears once)
    std::vector<size_t> cluster_pos;    // size M+1, prefix sums

    IVF_DocKMeans(size_t n_clusters, size_t d)
        : n_clusters(n_clusters), d(d) {}

    /**
     * Build the IVF from raw data using doc_kmeans clustering.
     *
     * @param data         All vectors, contiguous row-major (total_vectors * d)
     * @param doc_ptrs     Document boundaries, size N+1
     * @param N            Number of documents
     * @param total_vectors Total number of vectors across all documents
     * @param max_iter     Maximum K-means iterations
     */
    void build(const float* data, const int* doc_ptrs,
               size_t N, size_t total_vectors, size_t max_iter) {
        auto result = doc_kmeans(data, doc_ptrs, N, total_vectors, d, n_clusters, max_iter);

        centroids = std::move(result.centroids);

        // Build per-cluster doc ID lists
        std::vector<std::vector<int>> clusters(n_clusters);
        for (size_t i = 0; i < N; ++i) {
            int k = result.assignments[i];
            if (k >= 0 && static_cast<size_t>(k) < n_clusters) {
                clusters[k].push_back(static_cast<int>(i));
            }
        }

        // Flatten into inv_list + cluster_pos
        inv_list.clear();
        cluster_pos.clear();
        cluster_pos.reserve(n_clusters + 1);
        size_t cumu = 0;
        for (size_t k = 0; k < n_clusters; ++k) {
            cluster_pos.push_back(cumu);
            cumu += clusters[k].size();
            inv_list.insert(inv_list.end(), clusters[k].begin(), clusters[k].end());
        }
        cluster_pos.push_back(cumu);

        std::cout << ">>> IVF_DocKMeans built: " << N << " docs in " << n_clusters
                  << " clusters, inv_list size = " << inv_list.size() << std::endl;
    }

    int max_cluster_size() const {
        int max_size = 0;
        for (size_t i = 0; i < n_clusters; ++i) {
            int sz = static_cast<int>(cluster_pos[i + 1] - cluster_pos[i]);
            max_size = std::max(max_size, sz);
        }
        return max_size;
    }

    void save(const std::string& filename) const {
        std::ofstream of(filename + ".ivf_dk", std::ios::binary);
        of.write(reinterpret_cast<const char*>(&n_clusters), sizeof(size_t));
        of.write(reinterpret_cast<const char*>(&d), sizeof(size_t));
        // centroids
        of.write(reinterpret_cast<const char*>(centroids.data()), n_clusters * d * sizeof(float));
        // inv_list
        size_t inv_size = inv_list.size();
        of.write(reinterpret_cast<const char*>(&inv_size), sizeof(size_t));
        of.write(reinterpret_cast<const char*>(inv_list.data()), inv_size * sizeof(int));
        // cluster_pos
        of.write(reinterpret_cast<const char*>(cluster_pos.data()), (n_clusters + 1) * sizeof(size_t));
        of.close();
    }

    void load(const std::string& filename) {
        std::ifstream inf(filename + ".ivf_dk", std::ios::binary);
        inf.read(reinterpret_cast<char*>(&n_clusters), sizeof(size_t));
        inf.read(reinterpret_cast<char*>(&d), sizeof(size_t));
        // centroids
        centroids.resize(n_clusters * d);
        inf.read(reinterpret_cast<char*>(centroids.data()), n_clusters * d * sizeof(float));
        // inv_list
        size_t inv_size;
        inf.read(reinterpret_cast<char*>(&inv_size), sizeof(size_t));
        inv_list.resize(inv_size);
        inf.read(reinterpret_cast<char*>(inv_list.data()), inv_size * sizeof(int));
        // cluster_pos
        cluster_pos.resize(n_clusters + 1);
        inf.read(reinterpret_cast<char*>(cluster_pos.data()), (n_clusters + 1) * sizeof(size_t));
        inf.close();
        std::cout << ">>> IVF_DocKMeans loaded: " << n_clusters << " clusters, inv_list size = " << inv_list.size() << std::endl;
    }
};
