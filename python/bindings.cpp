#include "chimera/chimera_index.cuh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

using Chimera::SearchOptions;
using Chimera::chimera_index;
using FloatArray = py::array_t<float, py::array::c_style>;

SearchOptions search_options(
    int nprobe,
    int k_refine,
    int k_full_bit,
    int cagra_itopk_size,
    int num_chunks) {
    return {
        nprobe,
        k_refine,
        k_full_bit,
        cagra_itopk_size,
        num_chunks,
    };
}

std::unique_ptr<chimera_index> build(
    const FloatArray& embeddings,
    const std::vector<int>& doc_lens,
    size_t n_clusters,
    size_t ex_bits,
    int nprobe,
    int k_refine,
    int k_full_bit,
    int cagra_itopk_size,
    int num_chunks) {
    if (embeddings.ndim() != 2) {
        throw std::invalid_argument(
            "embeddings must have shape [num_embeddings, dimension]");
    }

    const auto values = embeddings.unchecked<2>();
    std::vector<float> data(
        embeddings.data(),
        embeddings.data() + embeddings.size());
    auto index = std::make_unique<chimera_index>();
    {
        py::gil_scoped_release release;
        index->build(
            data,
            static_cast<size_t>(values.shape(1)),
            doc_lens,
            n_clusters,
            ex_bits,
            search_options(
                nprobe,
                k_refine,
                k_full_bit,
                cagra_itopk_size,
                num_chunks));
    }
    return index;
}

std::unique_ptr<chimera_index> load(
    const std::string& index_dir,
    int nprobe,
    int k_refine,
    int k_full_bit,
    int cagra_itopk_size,
    int num_chunks) {
    auto index = std::make_unique<chimera_index>();
    {
        py::gil_scoped_release release;
        index->load(
            index_dir,
            search_options(
                nprobe,
                k_refine,
                k_full_bit,
                cagra_itopk_size,
                num_chunks));
    }
    return index;
}

py::object search(chimera_index& index, const FloatArray& queries, size_t k) {
    if (queries.ndim() != 2 && queries.ndim() != 3) {
        throw std::invalid_argument(
            "queries must have shape [query_length, dimension] or "
            "[num_queries, query_length, dimension]");
    }

    const size_t num_queries =
        queries.ndim() == 3 ? static_cast<size_t>(queries.shape(0)) : 1;
    const size_t query_length =
        static_cast<size_t>(queries.shape(queries.ndim() - 2));
    const size_t dimension =
        static_cast<size_t>(queries.shape(queries.ndim() - 1));
    const size_t query_size = query_length * dimension;
    const float* data = queries.data();
    std::vector<std::vector<size_t>> results(num_queries);
    {
        py::gil_scoped_release release;
        for (size_t i = 0; i < num_queries; ++i) {
            results[i] = index.search(
                data + i * query_size,
                query_size,
                k);
        }
    }

    if (queries.ndim() == 2) {
        return py::cast(std::move(results.front()));
    }
    return py::cast(std::move(results));
}

void save(const chimera_index& index, const std::string& index_dir) {
    py::gil_scoped_release release;
    index.save(index_dir);
}

}  // namespace

PYBIND11_MODULE(chimera, module) {
    const SearchOptions defaults;
    auto index = py::class_<chimera_index>(module, "ChimeraIndex");
    index.def_static(
            "build",
            &build,
            py::arg("embeddings").noconvert(),
            py::arg("doc_lens"),
            py::arg("n_clusters"),
            py::arg("ex_bits"),
            py::arg("nprobe") = defaults.nprobe,
            py::arg("k_refine") = defaults.k_refine,
            py::arg("k_full_bit") = defaults.k_full_bit,
            py::arg("cagra_itopk_size") = defaults.cagra_itopk_size,
            py::arg("num_chunks") = defaults.num_chunks)
        .def(
            "search",
            &search,
            py::arg("queries").noconvert(),
            py::arg("k") = 100)
        .def("save", &save, py::arg("index_dir"))
        .def_static(
            "load",
            &load,
            py::arg("index_dir"),
            py::arg("nprobe") = defaults.nprobe,
            py::arg("k_refine") = defaults.k_refine,
            py::arg("k_full_bit") = defaults.k_full_bit,
            py::arg("cagra_itopk_size") = defaults.cagra_itopk_size,
            py::arg("num_chunks") = defaults.num_chunks);
    if (py::hasattr(index, "_pybind11_conduit_v1_")) {
        py::delattr(index, "_pybind11_conduit_v1_");
    }
}
