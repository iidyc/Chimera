#include "arg_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/resources.hpp>

#include "rabitqlib/third/hnswlib/hnswlib.h"

namespace {

using cagra_index_t = cuvs::neighbors::cagra::index<float, uint32_t>;

void print_input_help(const char* program)
{
    std::cout
        << "Usage: " << program << "\n"
        << "  --input <centroids.carga>\n"
        << "  --output <hnsw_index.hnsw>\n"
        << "  [--M <graph_degree>]             Default: 16\n"
        << "  [--ef-construction <value>]      Default: 500\n"
        << "  [--batch-rows <rows>]            Default: 65536\n";
}

void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

int main(int argc, char** argv)
{
    std::string input_file;
    std::string output_file;
    int M = 16;
    int ef_construction = 500;
    size_t batch_rows = 65536;

    if (argc == 1) {
        print_input_help(argv[0]);
        return 1;
    }

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--input") {
                input_file = require_value(argc, argv, i, arg);
            } else if (arg == "--output") {
                output_file = require_value(argc, argv, i, arg);
            } else if (arg == "--M") {
                M = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--ef-construction") {
                ef_construction = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--batch-rows") {
                batch_rows = static_cast<size_t>(std::stoull(require_value(argc, argv, i, arg)));
            } else if (arg == "--help" || arg == "-h") {
                print_input_help(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    if (input_file.empty() || output_file.empty()) {
        std::cerr << "Missing required arguments.\n\n";
        print_input_help(argv[0]);
        return 1;
    }
    if (M <= 0 || ef_construction <= 0 || batch_rows == 0) {
        std::cerr << "M, ef-construction, and batch-rows must be > 0.\n";
        return 1;
    }

    raft::resources res;
    cagra_index_t cagra_index(res);
    cuvs::neighbors::cagra::deserialize(res, input_file, &cagra_index);

    const auto dataset = cagra_index.dataset();
    const size_t n_rows = static_cast<size_t>(dataset.extent(0));
    const size_t dim = static_cast<size_t>(dataset.extent(1));
    const size_t stride = static_cast<size_t>(dataset.stride(0));
    if (n_rows == 0 || dim == 0) {
        throw std::runtime_error("Deserialized CAGRA index has empty dataset");
    }

    std::cout
        << "Loaded CAGRA dataset from " << input_file
        << " with n=" << n_rows
        << " dim=" << dim
        << " stride=" << stride
        << std::endl;

    std::vector<float> host_data(n_rows * dim);
    size_t processed = 0;
    while (processed < n_rows) {
        const size_t rows = std::min(batch_rows, n_rows - processed);
        float* host_batch = host_data.data() + processed * dim;
        const float* dev_ptr = dataset.data_handle() + processed * stride;
        check_cuda(
            cudaMemcpy2D(
                host_batch,
                dim * sizeof(float),
                dev_ptr,
                stride * sizeof(float),
                dim * sizeof(float),
                rows,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy2D(dataset->host)");
        processed += rows;
        std::cout
            << "[HNSW] copied " << processed << "/" << n_rows
            << " centroids" << std::endl;
    }

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> hnsw(&space, n_rows, M, ef_construction);
#pragma omp parallel for
    for (size_t i = 0; i < n_rows; ++i) {
        hnsw.addPoint(host_data.data() + i * dim, i);
    }
    hnsw.saveIndex(output_file);
    std::cout
        << "Saved hnswlib HNSW index to " << output_file
        << " (M=" << M
        << ", efConstruction=" << ef_construction
        << ")" << std::endl;
    return 0;
}
