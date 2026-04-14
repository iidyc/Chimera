#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "rabitqlib/third/hnswlib/hnswlib.h"

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

// abstract class for PG index
struct PG {
    size_t n;
    size_t d;

    PG(size_t n, size_t d);

    virtual void build_index(const float* data) = 0;
    virtual void search(const float* query, size_t k, std::vector<size_t>& results) = 0;
    virtual void save(const std::string& filename) const = 0;
    virtual void load(const std::string& filename) = 0;
    virtual ~PG() = default;
};

struct PG_HNSW : PG {
    size_t M = 16;
    size_t ef_construction = 500;
    hnswlib::L2Space space_;
    hnswlib::HierarchicalNSW<float>* hnsw_index;

    PG_HNSW(size_t n, size_t d);
    ~PG_HNSW() override;

    void build_index(const float* data) override;
    void search(const float* query, size_t k, std::vector<size_t>& results) override;
    void save(const std::string& filename) const override;
    void load(const std::string& filename) override;
};

struct PG_CAGRA : PG {
    raft::resources res_;
    std::unique_ptr<cuvs::neighbors::cagra::index<float, uint32_t>> index_cagra;

    PG_CAGRA(size_t n, size_t d);
    ~PG_CAGRA() override = default;

    void build_index(const float* data) override;
    void search(const float* query, size_t k, std::vector<size_t>& results) override;

    // Batched search with GPU pointers.
    void search_batch_gpu(const float* d_queries, size_t n_queries, size_t k,
                          float* d_dists, uint32_t* d_labels, cudaStream_t stream);

    void save(const std::string& filename) const override;
    void load(const std::string& filename) override;
};

enum class PGType { HNSW, CAGRA };

struct IVF_PG {
    size_t n_clusters;
    size_t d;
    PG* pg_index;

    std::vector<uint32_t> inv_list;
    std::vector<size_t> cluster_pos;

    IVF_PG(size_t n_clusters, size_t d, PGType type = PGType::HNSW);
    ~IVF_PG();

    void search(const float* query, size_t n_probe, std::vector<size_t>& results);

    // Batched GPU search: delegates to PG_CAGRA::search_batch_gpu.
    void search_batch_gpu(const float* d_queries, size_t n_queries, size_t n_probe,
                          float* d_dists, uint32_t* d_labels, cudaStream_t stream);

    void build_index(const float* data);

    // Populate inv_list and cluster_pos from in-memory cluster assignments.
    void build_from_assignments(const uint32_t* list_nos, size_t n_vectors);

    void build_from_existing();

    int max_cluster_size() const;

    void save(const std::string& filename) const;
    void load(const std::string& filename);
    void save(const std::string& ivf_path, const std::string& graph_path) const;
    void load(const std::string& ivf_path, const std::string& graph_path);
};
