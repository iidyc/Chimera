#include "ivf_pg.hpp"

#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#ifdef CHIMERA_HAVE_CUVS
#include <rmm/cuda_stream_view.hpp>
#endif

namespace Chimera {

namespace {

#ifdef CHIMERA_HAVE_CUVS
using cagra_index_t = cuvs::neighbors::cagra::index<float, uint32_t>;
#endif

void save_graph_file(const PG* pg_index, const std::string& graph_path) {
#ifdef CHIMERA_HAVE_CUVS
    if (const auto* cagra = dynamic_cast<const PG_CAGRA*>(pg_index)) {
        cuvs::neighbors::cagra::serialize(cagra->res_, graph_path, *cagra->index_cagra);
        return;
    }
#endif

    if (const auto* hnsw = dynamic_cast<const PG_HNSW*>(pg_index)) {
        hnsw->hnsw_index->saveIndex(graph_path);
        return;
    }

    throw std::runtime_error("Unsupported PG type in save_graph_file()");
}

void load_graph_file(PG* pg_index, const std::string& graph_path) {
#ifdef CHIMERA_HAVE_CUVS
    if (auto* cagra = dynamic_cast<PG_CAGRA*>(pg_index)) {
        cagra->index_cagra = std::make_unique<cagra_index_t>(cagra->res_);
        cuvs::neighbors::cagra::deserialize(cagra->res_, graph_path, cagra->index_cagra.get());
        return;
    }
#endif

    if (auto* hnsw = dynamic_cast<PG_HNSW*>(pg_index)) {
        hnsw->hnsw_index->loadIndex(graph_path, &hnsw->space_, hnsw->n);
        return;
    }

    throw std::runtime_error("Unsupported PG type in load_graph_file()");
}

}  // namespace

// ---------------- PG ----------------

PG::PG(size_t n, size_t d) : n(n), d(d) {}

// ---------------- PG_HNSW ----------------

PG_HNSW::PG_HNSW(size_t n, size_t d) : PG(n, d), space_(d) {
    hnsw_index = new hnswlib::HierarchicalNSW<float>(&space_, n, M, ef_construction);
}

PG_HNSW::~PG_HNSW() {
    delete hnsw_index;
}

size_t PG_HNSW::search_ef_for_k(size_t k) {
    return std::clamp<size_t>(2 * k, 64, 768);
}

void PG_HNSW::build_index(const float* data) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        hnsw_index->addPoint(data + i * d, i);
    }
}

void PG_HNSW::search(const float* query, size_t k, std::vector<size_t>& results) {
    hnsw_index->setEf(search_ef_for_k(k));
    auto result = hnsw_index->searchKnn(query, k);
    while (!result.empty()) {
        results.push_back(result.top().second);
        result.pop();
    }
}

void PG_HNSW::save(const std::string& filename) const {
    hnsw_index->saveIndex(filename + ".hnsw");
}

void PG_HNSW::load(const std::string& filename) {
    hnsw_index->loadIndex(filename + ".hnsw", &space_, n);
}

// ---------------- PG_CAGRA ----------------
#ifdef CHIMERA_HAVE_CUVS
PG_CAGRA::PG_CAGRA(size_t n, size_t d) : PG(n, d) {}

void PG_CAGRA::build_index(const float* data) {
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

void PG_CAGRA::search(const float* /*query*/, size_t /*k*/, std::vector<size_t>& /*results*/) {}

void PG_CAGRA::search_batch_gpu(const float* d_queries, size_t n_queries, size_t k,
                                float* d_dists, uint32_t* d_labels, cudaStream_t stream,
                                size_t itopk_size) {
    search_batch_gpu(res_, d_queries, n_queries, k, d_dists, d_labels, stream, itopk_size);
}

void PG_CAGRA::search_batch_gpu(const raft::resources& res, const float* d_queries,
                                size_t n_queries, size_t k, float* d_dists,
                                uint32_t* d_labels, cudaStream_t stream,
                                size_t itopk_size) {
    raft::resource::set_cuda_stream(res, rmm::cuda_stream_view(stream));

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
    search_params.itopk_size = itopk_size;
    cuvs::neighbors::cagra::search(res, search_params, *index_cagra, queries, labels, distances);
}

void PG_CAGRA::save(const std::string& filename) const {
    cuvs::neighbors::cagra::serialize(res_, filename + ".cagra", *index_cagra);
}

void PG_CAGRA::load(const std::string& filename) {
    index_cagra = std::make_unique<cagra_index_t>(res_);
    cuvs::neighbors::cagra::deserialize(res_, filename + ".cagra", index_cagra.get());
}
#endif

// ---------------- IVF_PG ----------------

IVF_PG::IVF_PG(size_t n_clusters, size_t d, PGType type) : n_clusters(n_clusters), d(d) {
    if (type == PGType::CAGRA) {
#ifdef CHIMERA_HAVE_CUVS
        pg_index = new PG_CAGRA(n_clusters, d);
#else
        throw std::runtime_error("PG_CAGRA requires CHIMERA_HAVE_CUVS");
#endif
    } else {
        pg_index = new PG_HNSW(n_clusters, d);
    }
}

IVF_PG::~IVF_PG() {
    delete pg_index;
}

void IVF_PG::search(const float* query, size_t n_probe, std::vector<size_t>& results) {
    std::vector<size_t> cluster_ids;
    pg_index->search(query, n_probe, cluster_ids);
    for (size_t id : cluster_ids) {
        size_t start = cluster_pos[id];
        size_t end = cluster_pos[id + 1];
        for (size_t i = start; i < end; ++i) {
            results.push_back(static_cast<size_t>(inv_list[i]));
        }
    }
}

void IVF_PG::search_batch_gpu(const float* d_queries, size_t n_queries, size_t n_probe,
                              float* d_dists, uint32_t* d_labels, cudaStream_t stream,
                              size_t itopk_size) {
#ifdef CHIMERA_HAVE_CUVS
    auto* cagra = dynamic_cast<PG_CAGRA*>(pg_index);
    if (cagra) {
        cagra->search_batch_gpu(
            d_queries, n_queries, n_probe, d_dists, d_labels, stream, itopk_size);
    } else {
        throw std::runtime_error("search_batch_gpu requires PG_CAGRA");
    }
#else
    (void)d_queries;
    (void)n_queries;
    (void)n_probe;
    (void)d_dists;
    (void)d_labels;
    (void)stream;
    (void)itopk_size;
    throw std::runtime_error("search_batch_gpu requires CHIMERA_HAVE_CUVS");
#endif
}

#ifdef CHIMERA_HAVE_CUVS
void IVF_PG::search_batch_gpu(const raft::resources& res, const float* d_queries,
                              size_t n_queries, size_t n_probe, float* d_dists,
                              uint32_t* d_labels, cudaStream_t stream,
                              size_t itopk_size) {
    auto* cagra = dynamic_cast<PG_CAGRA*>(pg_index);
    if (cagra) {
        cagra->search_batch_gpu(
            res, d_queries, n_queries, n_probe, d_dists, d_labels, stream, itopk_size);
    } else {
        throw std::runtime_error("search_batch_gpu requires PG_CAGRA");
    }
}
#endif

void IVF_PG::build_index(const float* /*data*/) {}

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

void IVF_PG::build_from_existing() {
    std::ifstream list_no_in("list_nos.bin", std::ios::binary);
    int n;
    list_no_in.read((char*)&n, sizeof(int));
    std::vector<uint32_t> list_nos(n);
    list_no_in.read((char*)list_nos.data(), n * sizeof(uint32_t));

    std::vector<std::vector<uint32_t>> clusters(n_clusters);
    for (int i = 0; i < n; ++i) {
        clusters[list_nos[i]].push_back(static_cast<uint32_t>(i));
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

int IVF_PG::max_cluster_size() const {
    int max_size = 0;
    for (size_t i = 0; i < n_clusters; ++i) {
        max_size = std::max(max_size, (int)(cluster_pos[i + 1] - cluster_pos[i]));
    }
    return max_size;
}

void IVF_PG::save(const std::string& filename) const {
    save(filename + ".ivf", filename + ".ivf.cagra");
}

void IVF_PG::save(const std::string& ivf_path, const std::string& graph_path) const {
    save_graph_file(pg_index, graph_path);
    std::ofstream of(ivf_path, std::ios::binary);
    size_t n = inv_list.size();
    of.write((char*)&n, sizeof(size_t));
    of.write((char*)inv_list.data(), n * sizeof(uint32_t));
    of.write((char*)cluster_pos.data(), cluster_pos.size() * sizeof(size_t));
    of.close();
}

void IVF_PG::load(const std::string& filename) {
    load(filename + ".ivf", filename + ".ivf.cagra");
}

void IVF_PG::load(const std::string& ivf_path, const std::string& graph_path) {
    std::ifstream inf(ivf_path, std::ios::binary);
    size_t n;
    inf.read((char*)&n, sizeof(size_t));
    inv_list.resize(n);
    inf.read((char*)inv_list.data(), n * sizeof(uint32_t));
    cluster_pos.resize(n_clusters + 1);
    inf.read((char*)cluster_pos.data(), (n_clusters + 1) * sizeof(size_t));
    inf.close();

    load_graph_file(pg_index, graph_path);
}


}  // namespace Chimera
