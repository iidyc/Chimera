#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace Chimera {

struct PG_CAGRA {
    size_t n;
    size_t d;
    raft::resources res_;
    std::unique_ptr<cuvs::neighbors::cagra::index<float, uint32_t>> index_cagra;

    PG_CAGRA(size_t n, size_t d);

    void build_index(const float* data);
    void search_centroids(
        const float* d_queries,
        size_t n_queries,
        size_t nprobe,
        float* d_dists,
        uint32_t* d_labels,
        cudaStream_t stream,
        size_t cagra_itopk_size = 150);
    void search_centroids(
        const raft::resources& res,
        const float* d_queries,
        size_t n_queries,
        size_t nprobe,
        float* d_dists,
        uint32_t* d_labels,
        cudaStream_t stream,
        size_t cagra_itopk_size = 150);
    void save(const std::string& graph_path) const;
    void load(const std::string& graph_path);
};

struct IVF_PG {
    size_t n_clusters;
    size_t d;
    PG_CAGRA* pg_index;

    std::vector<uint32_t> inv_list;
    std::vector<size_t> cluster_pos;

    IVF_PG(size_t n_clusters, size_t d);
    ~IVF_PG();

    void search_centroids(
        const float* d_queries,
        size_t n_queries,
        size_t nprobe,
        float* d_dists,
        uint32_t* d_labels,
        cudaStream_t stream,
        size_t cagra_itopk_size = 150);
    void search_centroids(
        const raft::resources& res,
        const float* d_queries,
        size_t n_queries,
        size_t nprobe,
        float* d_dists,
        uint32_t* d_labels,
        cudaStream_t stream,
        size_t cagra_itopk_size = 150);

    void build_from_assignments(const uint32_t* list_nos, size_t n_vectors);
    int max_cluster_size() const;

    void save(const std::string& ivf_path, const std::string& graph_path) const;
    void load(const std::string& ivf_path, const std::string& graph_path);
};

}  // namespace Chimera
