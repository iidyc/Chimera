#include "chimera/ivf_pg.hpp"

#include <algorithm>
#include <fstream>

#include <rmm/cuda_stream_view.hpp>

namespace Chimera {

using cagra_index_t = cuvs::neighbors::cagra::index<float, uint32_t>;

PG_CAGRA::PG_CAGRA(size_t n, size_t d) : n(n), d(d) {}

void PG_CAGRA::build_index(const float* data) {
    auto dataset = raft::make_host_matrix_view(
        data,
        static_cast<std::int64_t>(n),
        static_cast<std::int64_t>(d));
    cuvs::neighbors::cagra::index_params index_params;
    index_cagra = std::make_unique<cagra_index_t>(
        cuvs::neighbors::cagra::build(res_, index_params, dataset));
}

void PG_CAGRA::search_centroids(
    const float* d_queries,
    size_t n_queries,
    size_t nprobe,
    float* d_dists,
    uint32_t* d_labels,
    cudaStream_t stream,
    size_t cagra_itopk_size) {
    search_centroids(
        res_, d_queries, n_queries, nprobe, d_dists, d_labels, stream,
        cagra_itopk_size);
}

void PG_CAGRA::search_centroids(
    const raft::resources& res,
    const float* d_queries,
    size_t n_queries,
    size_t nprobe,
    float* d_dists,
    uint32_t* d_labels,
    cudaStream_t stream,
    size_t cagra_itopk_size) {
    raft::resource::set_cuda_stream(res, rmm::cuda_stream_view(stream));

    auto queries = raft::make_device_matrix_view(
        d_queries,
        static_cast<std::int64_t>(n_queries),
        static_cast<std::int64_t>(d));
    auto distances = raft::make_device_matrix_view(
        d_dists,
        static_cast<std::int64_t>(n_queries),
        static_cast<std::int64_t>(nprobe));
    auto labels = raft::make_device_matrix_view(
        d_labels,
        static_cast<std::int64_t>(n_queries),
        static_cast<std::int64_t>(nprobe));

    cuvs::neighbors::cagra::search_params search_params;
    search_params.itopk_size = cagra_itopk_size;
    cuvs::neighbors::cagra::search(
        res, search_params, *index_cagra, queries, labels, distances);
}

void PG_CAGRA::save(const std::string& graph_path) const {
    cuvs::neighbors::cagra::serialize(res_, graph_path, *index_cagra);
}

void PG_CAGRA::load(const std::string& graph_path) {
    index_cagra = std::make_unique<cagra_index_t>(res_);
    cuvs::neighbors::cagra::deserialize(res_, graph_path, index_cagra.get());
}

IVF_PG::IVF_PG(size_t n_clusters, size_t d)
    : n_clusters(n_clusters), d(d), pg_index(new PG_CAGRA(n_clusters, d)) {}

IVF_PG::~IVF_PG() {
    delete pg_index;
}

void IVF_PG::search_centroids(
    const float* d_queries,
    size_t n_queries,
    size_t nprobe,
    float* d_dists,
    uint32_t* d_labels,
    cudaStream_t stream,
    size_t cagra_itopk_size) {
    pg_index->search_centroids(
        d_queries, n_queries, nprobe, d_dists, d_labels, stream,
        cagra_itopk_size);
}

void IVF_PG::search_centroids(
    const raft::resources& res,
    const float* d_queries,
    size_t n_queries,
    size_t nprobe,
    float* d_dists,
    uint32_t* d_labels,
    cudaStream_t stream,
    size_t cagra_itopk_size) {
    pg_index->search_centroids(
        res, d_queries, n_queries, nprobe, d_dists, d_labels, stream,
        cagra_itopk_size);
}

void IVF_PG::build_from_assignments(const uint32_t* list_nos, size_t n_vectors) {
    cluster_pos.assign(n_clusters + 1, 0);
    for (size_t i = 0; i < n_vectors; ++i) {
        ++cluster_pos[list_nos[i] + 1];
    }

    for (size_t i = 0; i < n_clusters; ++i) {
        cluster_pos[i + 1] += cluster_pos[i];
    }

    inv_list.resize(n_vectors);
    std::vector<size_t> write_pos = cluster_pos;
    for (size_t i = 0; i < n_vectors; ++i) {
        const auto cluster_id = list_nos[i];
        inv_list[write_pos[cluster_id]++] = static_cast<uint32_t>(i);
    }
}

int IVF_PG::max_cluster_size() const {
    int max_size = 0;
    for (size_t i = 0; i < n_clusters; ++i) {
        max_size = std::max(max_size, static_cast<int>(cluster_pos[i + 1] - cluster_pos[i]));
    }
    return max_size;
}

void IVF_PG::save(const std::string& ivf_path, const std::string& graph_path) const {
    pg_index->save(graph_path);

    std::ofstream output(ivf_path, std::ios::binary);
    const size_t n = inv_list.size();
    output.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
    output.write(
        reinterpret_cast<const char*>(inv_list.data()),
        static_cast<std::streamsize>(n * sizeof(uint32_t)));
    output.write(
        reinterpret_cast<const char*>(cluster_pos.data()),
        static_cast<std::streamsize>(cluster_pos.size() * sizeof(size_t)));
}

void IVF_PG::load(const std::string& ivf_path, const std::string& graph_path) {
    std::ifstream input(ivf_path, std::ios::binary);
    size_t n;
    input.read(reinterpret_cast<char*>(&n), sizeof(size_t));
    inv_list.resize(n);
    input.read(
        reinterpret_cast<char*>(inv_list.data()),
        static_cast<std::streamsize>(n * sizeof(uint32_t)));
    cluster_pos.resize(n_clusters + 1);
    input.read(
        reinterpret_cast<char*>(cluster_pos.data()),
        static_cast<std::streamsize>((n_clusters + 1) * sizeof(size_t)));

    pg_index->load(graph_path);
}

}  // namespace Chimera
