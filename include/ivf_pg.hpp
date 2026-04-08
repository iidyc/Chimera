#pragma once

#include <omp.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <stdexcept>

#include "rabitqlib/third/hnswlib/hnswlib.h"

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace {

using cagra_index_t = cuvs::neighbors::cagra::index<float, uint32_t>;

}  // namespace

// abstract class for PG index
struct PG {
    size_t n;
    size_t d;

    PG(size_t n, size_t d) : n(n), d(d) {}

    virtual void build_index(const float* data) = 0;
    virtual void search(const float* query, size_t k, std::vector<size_t>& results) = 0;
    virtual void save(const std::string& filename) const = 0;
    virtual void load(const std::string& filename) = 0;
    virtual ~PG() = default;
};

struct PG_HNSW: PG {
    size_t M = 16;
    size_t ef_construction = 500;
    hnswlib::L2Space space_;
    hnswlib::HierarchicalNSW<float>* hnsw_index;

    PG_HNSW(size_t n, size_t d): PG(n, d), space_(d) {
        hnsw_index = new hnswlib::HierarchicalNSW<float>(&space_, n, M, ef_construction);
    }

    void build_index(const float* data) override {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            hnsw_index->addPoint(data + i * d, i);
        }
    }

    void search(const float* query, size_t k, std::vector<size_t>& results) override {
        hnsw_index->setEf(std::max(768UL, 2 * k));
        auto result = hnsw_index->searchKnn(query, k);
        while (!result.empty()) {
            results.push_back(result.top().second);
            result.pop();
        }
    }

    void save(const std::string& filename) const override {
        hnsw_index->saveIndex(filename + ".hnsw");
    }

    void load(const std::string& filename) override {
        hnsw_index->loadIndex(filename + ".hnsw", &space_, n);
    }

    ~PG_HNSW() {
        delete hnsw_index;
    }
};

struct PG_CAGRA: PG {
    raft::resources res_;
    // raft::resources search_res_;
    std::unique_ptr<cuvs::neighbors::cagra::index<float, uint32_t>> index_cagra;

    PG_CAGRA(size_t n, size_t d): PG(n, d) {}

    void build_index(const float* data) override {
        auto dataset = raft::make_host_matrix_view(
            data,
            static_cast<std::int64_t>(n),
            static_cast<std::int64_t>(d)
        );
        cuvs::neighbors::cagra::index_params index_params;
        index_cagra = std::make_unique<cagra_index_t>(
            cuvs::neighbors::cagra::build(res_, index_params, dataset)
        );
    }

    void search(const float* query, size_t k, std::vector<size_t>& results) override {}

    // Batched search with GPU pointers.
    void search_batch_gpu(const float* d_queries, size_t n_queries, size_t k,
                          float* d_dists, uint32_t* d_labels, cudaStream_t stream) {
        // raft::resource::set_cuda_stream(res_, rmm::cuda_stream_view(stream));
        auto queries = raft::make_device_matrix_view(
            d_queries,
            static_cast<std::int64_t>(n_queries),
            static_cast<std::int64_t>(d)
        );
        auto distances = raft::make_device_matrix_view(
            d_dists,
            static_cast<std::int64_t>(n_queries),
            static_cast<std::int64_t>(k)
        );
        auto labels = raft::make_device_matrix_view(
            d_labels,
            static_cast<std::int64_t>(n_queries),
            static_cast<std::int64_t>(k)
        );

        cuvs::neighbors::cagra::search_params search_params;
        search_params.itopk_size = 150;
        cuvs::neighbors::cagra::search(res_, search_params, *index_cagra, queries, labels, distances);
    }

    void save(const std::string& filename) const override {
        cuvs::neighbors::cagra::serialize(res_, filename + ".cagra", *index_cagra);
    }

    void load(const std::string& filename) override {
        index_cagra = std::make_unique<cagra_index_t>(res_);
        cuvs::neighbors::cagra::deserialize(res_, filename + ".cagra", index_cagra.get());
    }

    ~PG_CAGRA() = default;
};

enum class PGType { HNSW, CAGRA };

struct IVF_PG {
    size_t n_clusters;
    size_t d;
    PG* pg_index;

    std::vector<int> inv_list;
    std::vector<size_t> cluster_pos;

    IVF_PG(size_t n_clusters, size_t d, PGType type = PGType::HNSW): n_clusters(n_clusters), d(d) {
        if (type == PGType::CAGRA) {
            pg_index = new PG_CAGRA(n_clusters, d);
        } else {
            pg_index = new PG_HNSW(n_clusters, d);
        }
    }

    void search(const float* query, size_t n_probe, std::vector<size_t>& results) {
        std::vector<size_t> cluster_ids;
        pg_index->search(query, n_probe, cluster_ids);
        for (size_t id : cluster_ids) {
            size_t start = cluster_pos[id];
            size_t end = cluster_pos[id + 1];
            for (size_t i = start; i < end; ++i) {
                results.push_back(inv_list[i]);
            }
        }
    }

    // Batched GPU search: delegates to PG_CAGRA::search_batch_gpu.
    // All pointers can be GPU-resident.
    void search_batch_gpu(const float* d_queries, size_t n_queries, size_t n_probe,
                          float* d_dists, uint32_t* d_labels, cudaStream_t stream) {
        auto* cagra = dynamic_cast<PG_CAGRA*>(pg_index);
        if (cagra) {
            cagra->search_batch_gpu(d_queries, n_queries, n_probe, d_dists, d_labels, stream);
        } else {
            throw std::runtime_error("search_batch_gpu requires PG_CAGRA");
        }
    }

    void build_index(const float* data) {

    }

    // Populate inv_list and cluster_pos from in-memory cluster assignments.
    // list_nos[i] is the cluster ID for embedding i, i in [0, n_vectors).
    void build_from_assignments(const int64_t* list_nos, size_t n_vectors) {
        std::vector<std::vector<int>> clusters(n_clusters);
        for (size_t i = 0; i < n_vectors; ++i) {
            clusters[list_nos[i]].push_back(static_cast<int>(i));
        }
        inv_list.clear();
        cluster_pos.clear();
        cluster_pos.reserve(n_clusters + 1);
        size_t cumu_size = 0;
        for (size_t i = 0; i < n_clusters; ++i) {
            cluster_pos.push_back(cumu_size);
            cumu_size += clusters[i].size();
            inv_list.insert(inv_list.end(), clusters[i].begin(), clusters[i].end());
        }
        cluster_pos.push_back(cumu_size);
    }

    void build_from_existing() {
        std::ifstream list_no_in("list_nos.bin", std::ios::binary);
        int n;
        list_no_in.read((char*)&n, sizeof(int));
        std::vector<int> list_nos(n);
        list_no_in.read((char*)list_nos.data(), n * sizeof(int));

        std::vector<std::vector<int>> clusters(n_clusters);
        for (int i = 0; i < n; ++i) {
            clusters[list_nos[i]].push_back(i);
        }
        size_t cumu_size = 0;
        for (size_t i = 0; i < n_clusters; ++i) {
            cluster_pos.push_back(cumu_size);
            cumu_size += clusters[i].size();
            inv_list.insert(inv_list.end(), clusters[i].begin(), clusters[i].end());
        }
        cluster_pos.push_back(cumu_size);

        pg_index->load("index");
    }

    int max_cluster_size() const {
        int max_size = 0;
        for (size_t i = 0; i < n_clusters; ++i) {
            max_size = std::max(max_size, (int)(cluster_pos[i + 1] - cluster_pos[i]));
        }
        return max_size;
    }

    void save(const std::string& filename) const {
        pg_index->save(filename + ".ivf");
        std::ofstream of(filename + ".ivf", std::ios::binary);
        size_t n = inv_list.size();
        of.write((char*)&n, sizeof(size_t));
        of.write((char*)inv_list.data(), n * sizeof(int));
        of.write((char*)cluster_pos.data(), cluster_pos.size() * sizeof(size_t));
        of.close();
    }

    void load(const std::string& filename) {
        std::ifstream inf(filename + ".ivf", std::ios::binary);
        size_t n;
        inf.read((char*)&n, sizeof(size_t));
        inv_list.resize(n);
        inf.read((char*)inv_list.data(), n * sizeof(int));
        cluster_pos.resize(n_clusters + 1);
        inf.read((char*)cluster_pos.data(), (n_clusters + 1) * sizeof(size_t));
        inf.close();

        pg_index->load(filename + ".ivf");
    }
    
    ~IVF_PG() {
        delete pg_index;
    }
};
