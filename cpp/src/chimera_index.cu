#include <cuda_runtime.h>

#include "chimera/chimera_index.cuh"

#define EIGEN_NO_CUDA

#include "chimera/config.cuh"
#include "chimera/io.hpp"
#include "chimera/ivf_pg.hpp"
#include "chimera/quantization.hpp"
#include "chimera_index_internal.cuh"
#include "rabitqlib/utils/rotator.hpp"

#include <raft/core/resources.hpp>

#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace Chimera {

namespace {

void validate_search_options(const SearchOptions& options) {
    if (options.nprobe <= 0 || options.k_refine <= 0 ||
        options.k_full_bit <= 0 || options.cagra_itopk_size <= 0 ||
        options.num_chunks <= 0) {
        throw std::invalid_argument("All Chimera search options must be positive");
    }
    if (options.k_full_bit > options.k_refine) {
        throw std::invalid_argument("k_full_bit must not exceed k_refine");
    }
}

}  // namespace

chimera_index::chimera_index() = default;

void chimera_index::begin_initialization(const SearchOptions& options) {
    release_resources();
    options_ = options;
    validate_search_options(options_);
    cagra_res_ = std::make_unique<raft::resources>();
    ws_ = std::make_unique<Workspace>();
}

chimera_index::~chimera_index() {
    release_resources();
}

void chimera_index::load(
    const std::string& index_dir,
    const SearchOptions& options) {
    const std::filesystem::path directory(index_dir);
    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error(
            "Chimera index directory does not exist: " + index_dir);
    }

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "Querying CUDA devices");
    if (device_count == 0) {
        throw std::runtime_error("No CUDA device is available");
    }
    check_cuda(cudaSetDevice(0), "Selecting CUDA device 0");
    begin_initialization(options);

    const auto doclens_path = (directory / "doclens.bin").string();
    const auto doc_one_bit_path = (directory / "doc_1bit.bin").string();
    const auto doc_full_path = (directory / "doc_full.bin").string();
    const auto cluster_one_bit_path =
        (directory / "cluster_1bit.bin").string();
    const auto ivf_path = (directory / "ivf.bin").string();
    const auto centroids_path = (directory / "centroids.carga").string();

    std::ifstream inf(doc_one_bit_path, std::ios::binary);
    IndexHeader header {};
    inf.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!inf || header.num_embeddings == 0 ||
        header.num_embeddings > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        header.dimension < 64 || header.dimension > PADDED_DIM ||
        header.num_clusters == 0 ||
        header.num_clusters > header.num_embeddings ||
        header.ex_bits < 1 || header.ex_bits > 7) {
        throw std::runtime_error("Invalid Chimera index header: " + doc_one_bit_path);
    }
    n = header.num_embeddings;
    d = header.dimension;
    n_clusters = header.num_clusters;
    ex_bits = header.ex_bits;
    if (static_cast<size_t>(options_.nprobe) > n_clusters) {
        throw std::invalid_argument("nprobe must not exceed the number of clusters");
    }

    if (header.padded_dimension != PADDED_DIM) {
        throw std::runtime_error(
            "Index file padded_dim=" +
            std::to_string(header.padded_dimension) +
            " does not match compiled PADDED_DIM=" + std::to_string(PADDED_DIM) +
            ". Please recompile with matching PADDED_DIM in chimera/config.cuh"
        );
    }

    rotator_ = rabitqlib::choose_rotator<float>(
        d,
        rabitqlib::RotatorType::FhtKacRotator,
        header.padded_dimension);
    rotator_->load(inf);
    if (!inf) {
        delete rotator_;
        rotator_ = nullptr;
        throw std::runtime_error("Failed to read rotator from " + doc_one_bit_path);
    }

    one_bit_codes_.resize(n * (PADDED_DIM / 64));
    one_bit_factors_.resize(n);
    full_bit_codes_.resize(n * PADDED_DIM * (1 + ex_bits) / 8);
    full_bit_factors_.resize(n);

    inf.read(
        reinterpret_cast<char*>(one_bit_codes_.data()),
        static_cast<std::streamsize>(one_bit_codes_.size() * sizeof(uint64_t)));
    inf.read(
        reinterpret_cast<char*>(one_bit_factors_.data()),
        static_cast<std::streamsize>(one_bit_factors_.size() * sizeof(float)));
    inf.read(
        reinterpret_cast<char*>(full_bit_factors_.data()),
        static_cast<std::streamsize>(full_bit_factors_.size() * sizeof(float)));
    if (!inf) {
        throw std::runtime_error(
            "Failed to read quantized payload from " + doc_one_bit_path);
    }
    std::ifstream full_bit_codes_file(doc_full_path, std::ios::binary);
    full_bit_codes_file.read(full_bit_codes_.data(), full_bit_codes_.size());
    if (!full_bit_codes_file) {
        throw std::runtime_error(
            "Failed to read full-code payload from " + doc_full_path);
    }
    ivf = new IVF_PG(n_clusters, d);
    ivf->load(ivf_path, centroids_path);
    doc_lens_ = load_doclens(doclens_path);

    const size_t clustered_bytes =
        n * (PADDED_DIM / 8 + sizeof(float) + sizeof(int) + sizeof(uint32_t));
    std::ifstream clustered_file(
        cluster_one_bit_path,
        std::ios::binary | std::ios::ate);
    if (!clustered_file || clustered_file.tellg() < 0 ||
        static_cast<size_t>(clustered_file.tellg()) != clustered_bytes) {
        throw std::runtime_error(
            "Invalid clustered index payload: " + cluster_one_bit_path);
    }
    clustered_data_.resize(clustered_bytes);
    clustered_file.seekg(0);
    clustered_file.read(
        clustered_data_.data(),
        static_cast<std::streamsize>(clustered_data_.size()));
    if (!clustered_file) {
        throw std::runtime_error(
            "Failed to read clustered index: " + cluster_one_bit_path);
    }

    has_index_ = true;
    try {
        initialize_search_state();
    } catch (...) {
        release_resources();
        throw;
    }
}

void chimera_index::initialize_search_state() {
    if (search_ready_) {
        return;
    }
    if (!has_index_ || ivf == nullptr || rotator_ == nullptr || ws_ == nullptr) {
        throw std::runtime_error("Cannot initialize an empty Chimera index");
    }
    if (static_cast<size_t>(options_.nprobe) > n_clusters) {
        throw std::invalid_argument("nprobe must not exceed the number of clusters");
    }

    unpack_func_ = rabitqlib::select_excode_unpackfunc(1 + ex_bits);
    max_cluster_size = ivf->max_cluster_size();
    set_doc_mapping(doc_lens_);

    check_cuda(cudaMalloc(&d_doc_ptrs_, (num_docs + 1) * sizeof(int)));
    check_cuda(cudaMalloc(&d_cluster_pos_, (ivf->n_clusters + 1) * sizeof(size_t)));
    check_cuda(cudaMemcpy(
        d_cluster_pos_,
        ivf->cluster_pos.data(),
        (ivf->n_clusters + 1) * sizeof(size_t),
        cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(
        d_doc_ptrs_,
        doc_ptrs_.data(),
        (num_docs + 1) * sizeof(int),
        cudaMemcpyHostToDevice));

    const size_t code_per_vec = PADDED_DIM / 8;
    const size_t expected_clustered_bytes =
        n * (code_per_vec + sizeof(float) + sizeof(int) + sizeof(uint32_t));
    if (clustered_data_.size() != expected_clustered_bytes) {
        throw std::runtime_error("Invalid in-memory clustered index payload");
    }

    size_t free_mem = 0;
    [[maybe_unused]] size_t total_mem = 0;
    check_cuda(cudaMemGetInfo(&free_mem, &total_mem));
    if (free_mem <= clustered_data_.size()) {
        throw std::runtime_error(
            "Insufficient GPU memory for clustered index: need " +
            std::to_string(clustered_data_.size()) + " bytes, have " +
            std::to_string(free_mem) + " bytes free");
    }

    const char* clustered_code = clustered_data_.data();
    const auto* clustered_factor = reinterpret_cast<const float*>(
        clustered_code + n * code_per_vec);
    const auto* clustered_doc_ids = reinterpret_cast<const int*>(
        reinterpret_cast<const char*>(clustered_factor) + n * sizeof(float));
    const auto* token_to_cluster_pos = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const char*>(clustered_doc_ids) + n * sizeof(int));

    check_cuda(cudaMalloc(&d_clustered_one_bit_codes_, n * code_per_vec));
    check_cuda(cudaMalloc(&d_clustered_one_bit_factors_, n * sizeof(float)));
    check_cuda(cudaMalloc(&d_clustered_doc_ids_, n * sizeof(int)));
    check_cuda(cudaMalloc(&d_token_to_cluster_pos_, n * sizeof(uint32_t)));
    check_cuda(cudaMemcpy(
        d_clustered_one_bit_codes_, clustered_code,
        n * code_per_vec, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(
        d_clustered_one_bit_factors_, clustered_factor,
        n * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(
        d_clustered_doc_ids_, clustered_doc_ids,
        n * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(
        d_token_to_cluster_pos_, token_to_cluster_pos,
        n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    compute_candidate_bounds();
    allocate_workspace();
    search_ready_ = true;
}

}  // namespace Chimera
