#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "chimera/chimera_index.cuh"

#define EIGEN_NO_CUDA

#include <omp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "chimera/config.cuh"
#include "chimera/ivf_pg.hpp"
#include "chimera/quantization.hpp"
#include "rabitqlib/utils/rotator.hpp"

namespace Chimera {

using namespace rabitqlib;

namespace {

constexpr std::uint64_t kCentroidSampleSeed = 42;
constexpr std::uint64_t kRotatorSeed = 42;
constexpr size_t kAssignmentBatchSize = 65536;
constexpr RotatorType kRotatorType = RotatorType::FhtKacRotator;

template <typename T>
class DeviceBuffer {
   public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                "cudaMalloc");
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    T* data() { return data_; }

   private:
    T* data_ = nullptr;
};

size_t checked_product(size_t lhs, size_t rhs, const char* description) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::runtime_error(
            std::string("Size overflow while allocating ") + description);
    }
    return lhs * rhs;
}

size_t validate_build_inputs(
    const std::vector<float>& embeddings,
    size_t dimension,
    const std::vector<int>& doc_lens,
    size_t n_clusters,
    size_t ex_bits) {
    if (dimension < 64 || dimension > PADDED_DIM) {
        throw std::invalid_argument(
            "Embedding dimension must be in [64, " +
            std::to_string(PADDED_DIM) + "]");
    }
    if (embeddings.empty() || embeddings.size() % dimension != 0) {
        throw std::invalid_argument(
            "Embedding storage must contain a non-empty whole number of vectors");
    }

    const size_t num_embeddings = embeddings.size() / dimension;
    if (num_embeddings > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "Embedding count exceeds the int32 index format limit");
    }
    if (n_clusters == 0 || n_clusters > num_embeddings) {
        throw std::invalid_argument(
            "n_clusters must be between 1 and the number of embeddings");
    }
    if (ex_bits < 1 || ex_bits > 7) {
        throw std::invalid_argument(
            "ex_bits must be in [1, 7] so full codes fit in 2-8 bits per dimension");
    }
    if (doc_lens.empty()) {
        throw std::invalid_argument("doc_lens must not be empty");
    }

    size_t token_count = 0;
    for (const int doc_len : doc_lens) {
        if (doc_len <= 0) {
            throw std::invalid_argument("Every document length must be positive");
        }
        const size_t length = static_cast<size_t>(doc_len);
        if (token_count > num_embeddings || length > num_embeddings - token_count) {
            throw std::invalid_argument(
                "Document lengths exceed the number of embeddings");
        }
        token_count += length;
    }
    if (token_count != num_embeddings) {
        throw std::invalid_argument(
            "Document lengths do not sum to the number of embeddings");
    }

    return num_embeddings;
}

void prepare_output_directory(const std::string& index_dir) {
    std::error_code error;
    std::filesystem::create_directories(index_dir, error);
    if (error) {
        throw std::runtime_error(
            "Failed to create index directory " + index_dir + ": " +
            error.message());
    }

    const std::array<const char*, 6> filenames = {
        "doclens.bin",
        "doc_1bit.bin",
        "doc_full.bin",
        "cluster_1bit.bin",
        "ivf.bin",
        "centroids.carga",
    };

    for (const char* filename : filenames) {
        const auto output =
            (std::filesystem::path(index_dir) / filename).string();
        error.clear();
        std::filesystem::remove(output, error);
        if (error) {
            throw std::runtime_error(
                "Failed to remove stale index output " + output + ": " +
                error.message());
        }
    }
}

std::vector<float> sample_centroids(
    const std::vector<float>& embeddings,
    size_t num_embeddings,
    size_t dimension,
    size_t n_clusters) {
    std::vector<size_t> sampled_ids(n_clusters);
    std::iota(sampled_ids.begin(), sampled_ids.end(), size_t {0});

    std::mt19937_64 generator(kCentroidSampleSeed);
    for (size_t index = n_clusters; index < num_embeddings; ++index) {
        std::uniform_int_distribution<size_t> distribution(0, index);
        const size_t position = distribution(generator);
        if (position < n_clusters) {
            sampled_ids[position] = index;
        }
    }

    std::vector<float> centroids(checked_product(n_clusters, dimension, "centroids"));
    for (size_t cluster = 0; cluster < n_clusters; ++cluster) {
        std::memcpy(
            centroids.data() + cluster * dimension,
            embeddings.data() + sampled_ids[cluster] * dimension,
            dimension * sizeof(float));
    }
    return centroids;
}

std::vector<uint32_t> assign_embeddings(
    const std::vector<float>& embeddings,
    size_t num_embeddings,
    size_t dimension,
    const std::vector<float>& centroids,
    size_t n_clusters) {
    PG_CAGRA assignment_index(n_clusters, dimension);
    assignment_index.build_index(centroids.data());

    const size_t batch_capacity = std::min(num_embeddings, kAssignmentBatchSize);
    DeviceBuffer<float> device_embeddings(
        checked_product(batch_capacity, dimension, "assignment batch"));
    DeviceBuffer<uint32_t> device_labels(batch_capacity);
    DeviceBuffer<float> device_distances(batch_capacity);

    std::vector<uint32_t> assignments(num_embeddings);
    for (size_t start = 0; start < num_embeddings; start += batch_capacity) {
        const size_t count = std::min(batch_capacity, num_embeddings - start);
        check_cuda(
            cudaMemcpy(
                device_embeddings.data(),
                embeddings.data() + start * dimension,
                count * dimension * sizeof(float),
                cudaMemcpyHostToDevice),
            "Copying embeddings to the GPU");

        assignment_index.search_centroids(
            device_embeddings.data(),
            count,
            1,
            device_distances.data(),
            device_labels.data(),
            nullptr);

        check_cuda(
            cudaMemcpy(
                assignments.data() + start,
                device_labels.data(),
                count * sizeof(uint32_t),
                cudaMemcpyDeviceToHost),
            "Copying centroid assignments to the host");

    }
    return assignments;
}

std::vector<float> rotate_centroids(
    const std::vector<float>& centroids,
    size_t n_clusters,
    size_t dimension,
    const Rotator<float>& rotator) {
    std::vector<float> rotated(
        checked_product(n_clusters, static_cast<size_t>(PADDED_DIM), "rotated centroids"));

#pragma omp parallel for schedule(static)
    for (std::int64_t cluster = 0;
         cluster < static_cast<std::int64_t>(n_clusters);
         ++cluster) {
        rotator.rotate(
            centroids.data() + static_cast<size_t>(cluster) * dimension,
            rotated.data() + static_cast<size_t>(cluster) * PADDED_DIM);
    }
    return rotated;
}

struct EncodedDocuments {
    std::vector<uint64_t> one_bit_codes;
    std::vector<uint8_t> full_codes;
    std::vector<float> one_bit_factors;
    std::vector<float> full_code_factors;
};

EncodedDocuments encode_embeddings(
    const std::vector<float>& embeddings,
    size_t num_embeddings,
    size_t dimension,
    size_t ex_bits,
    const Rotator<float>& rotator) {
    const size_t one_bit_words_per_vector = PADDED_DIM / 64;
    const size_t full_code_bytes_per_vector = PADDED_DIM * (1 + ex_bits) / 8;

    EncodedDocuments encoded {
        std::vector<uint64_t>(checked_product(
            num_embeddings, one_bit_words_per_vector, "one-bit codes")),
        std::vector<uint8_t>(checked_product(
            num_embeddings, full_code_bytes_per_vector, "full codes")),
        std::vector<float>(num_embeddings),
        std::vector<float>(num_embeddings),
    };

#pragma omp parallel
    {
        std::vector<float> rotated(PADDED_DIM);

#pragma omp for schedule(static)
        for (std::int64_t row = 0;
             row < static_cast<std::int64_t>(num_embeddings);
             ++row) {
            const size_t index = static_cast<size_t>(row);
            rotator.rotate(
                embeddings.data() + index * dimension,
                rotated.data());
            encode_one_bit(
                rotated.data(),
                PADDED_DIM,
                encoded.one_bit_codes.data() + index * one_bit_words_per_vector,
                &encoded.one_bit_factors[index]);
            encode_full_code(
                rotated.data(),
                PADDED_DIM,
                ex_bits,
                encoded.full_codes.data() + index * full_code_bytes_per_vector,
                &encoded.full_code_factors[index]);
        }
    }

    return encoded;
}

std::vector<int> make_token_doc_ids(
    const std::vector<int>& doc_lens,
    size_t num_embeddings) {
    std::vector<int> token_doc_ids(num_embeddings);
    size_t token = 0;
    for (size_t doc_id = 0; doc_id < doc_lens.size(); ++doc_id) {
        const size_t next = token + static_cast<size_t>(doc_lens[doc_id]);
        std::fill(
            token_doc_ids.begin() + static_cast<std::ptrdiff_t>(token),
            token_doc_ids.begin() + static_cast<std::ptrdiff_t>(next),
            static_cast<int>(doc_id));
        token = next;
    }
    return token_doc_ids;
}

struct ClusteredDocuments {
    std::vector<char> one_bit_codes;
    std::vector<float> one_bit_factors;
    std::vector<int> doc_ids;
    std::vector<uint32_t> token_to_cluster_position;
};

ClusteredDocuments reorder_by_cluster(
    const EncodedDocuments& encoded,
    const IVF_PG& ivf,
    const std::vector<int>& token_doc_ids) {
    const size_t num_embeddings = ivf.inv_list.size();
    const size_t code_bytes_per_vector = PADDED_DIM / 8;
    ClusteredDocuments clustered {
        std::vector<char>(checked_product(
            num_embeddings, code_bytes_per_vector, "clustered one-bit codes")),
        std::vector<float>(num_embeddings),
        std::vector<int>(num_embeddings),
        std::vector<uint32_t>(num_embeddings),
    };

    const auto* source_codes =
        reinterpret_cast<const char*>(encoded.one_bit_codes.data());
#pragma omp parallel for schedule(static)
    for (std::int64_t position = 0;
         position < static_cast<std::int64_t>(num_embeddings);
         ++position) {
        const size_t cluster_position = static_cast<size_t>(position);
        const uint32_t token_id = ivf.inv_list[cluster_position];
        std::memcpy(
            clustered.one_bit_codes.data() +
                cluster_position * code_bytes_per_vector,
            source_codes + static_cast<size_t>(token_id) * code_bytes_per_vector,
            code_bytes_per_vector);
        clustered.one_bit_factors[cluster_position] =
            encoded.one_bit_factors[token_id];
        clustered.doc_ids[cluster_position] = token_doc_ids[token_id];
        clustered.token_to_cluster_position[token_id] =
            static_cast<uint32_t>(cluster_position);
    }

    return clustered;
}

std::ofstream open_binary_output(const std::string& path) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    return output;
}

void finish_output(std::ofstream& output, const std::string& path) {
    output.flush();
    if (!output) {
        throw std::runtime_error("Failed to write output file: " + path);
    }
}

void write_doc_lens(
    const std::vector<int>& doc_lens,
    const std::string& path) {
    auto output = open_binary_output(path);
    const int count = static_cast<int>(doc_lens.size());
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    output.write(
        reinterpret_cast<const char*>(doc_lens.data()),
        static_cast<std::streamsize>(doc_lens.size() * sizeof(int)));
    finish_output(output, path);
}

}  // namespace

void chimera_index::build(
    const std::vector<float>& embeddings,
    size_t dimension,
    const std::vector<int>& doc_lens,
    size_t n_clusters,
    size_t ex_bits,
    const SearchOptions& options) {
    const size_t num_embeddings = validate_build_inputs(
        embeddings,
        dimension,
        doc_lens,
        n_clusters,
        ex_bits);

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "Querying CUDA devices");
    if (device_count == 0) {
        throw std::runtime_error("No CUDA device is available");
    }
    check_cuda(cudaSetDevice(0), "Selecting CUDA device 0");

    begin_initialization(options);
    n = num_embeddings;
    d = dimension;
    this->n_clusters = n_clusters;
    this->ex_bits = ex_bits;
    doc_lens_ = doc_lens;

    auto centroids = sample_centroids(
        embeddings,
        num_embeddings,
        dimension,
        n_clusters);
    auto assignments = assign_embeddings(
        embeddings,
        num_embeddings,
        dimension,
        centroids,
        n_clusters);
    auto rotator = std::unique_ptr<Rotator<float>>(
        choose_rotator_with_seed<float>(
            dimension,
            kRotatorType,
            PADDED_DIM,
            kRotatorSeed));

    auto rotated_centroids = rotate_centroids(
        centroids,
        n_clusters,
        dimension,
        *rotator);
    auto centroid_graph = std::make_unique<PG_CAGRA>(n_clusters, PADDED_DIM);
    centroid_graph->build_index(rotated_centroids.data());

    ivf = new IVF_PG(n_clusters, dimension);
    delete ivf->pg_index;
    ivf->pg_index = centroid_graph.release();
    ivf->build_from_assignments(assignments.data(), num_embeddings);

    auto encoded = encode_embeddings(
        embeddings,
        num_embeddings,
        dimension,
        ex_bits,
        *rotator);
    const auto token_doc_ids = make_token_doc_ids(doc_lens, num_embeddings);
    auto clustered = reorder_by_cluster(encoded, *ivf, token_doc_ids);

    rotator_ = rotator.release();
    one_bit_codes_ = std::move(encoded.one_bit_codes);
    one_bit_factors_ = std::move(encoded.one_bit_factors);
    full_bit_factors_ = std::move(encoded.full_code_factors);
    full_bit_codes_.assign(
        reinterpret_cast<const char*>(encoded.full_codes.data()),
        reinterpret_cast<const char*>(encoded.full_codes.data() + encoded.full_codes.size()));

    const size_t clustered_code_bytes = clustered.one_bit_codes.size();
    const size_t clustered_factor_bytes =
        clustered.one_bit_factors.size() * sizeof(float);
    const size_t clustered_doc_id_bytes = clustered.doc_ids.size() * sizeof(int);
    const size_t clustered_position_bytes =
        clustered.token_to_cluster_position.size() * sizeof(uint32_t);
    clustered_data_.resize(
        clustered_code_bytes + clustered_factor_bytes +
        clustered_doc_id_bytes + clustered_position_bytes);
    char* clustered_output = clustered_data_.data();
    std::memcpy(
        clustered_output,
        clustered.one_bit_codes.data(),
        clustered_code_bytes);
    clustered_output += clustered_code_bytes;
    std::memcpy(
        clustered_output,
        clustered.one_bit_factors.data(),
        clustered_factor_bytes);
    clustered_output += clustered_factor_bytes;
    std::memcpy(
        clustered_output,
        clustered.doc_ids.data(),
        clustered_doc_id_bytes);
    clustered_output += clustered_doc_id_bytes;
    std::memcpy(
        clustered_output,
        clustered.token_to_cluster_position.data(),
        clustered_position_bytes);

    has_index_ = true;
}

void chimera_index::save(const std::string& index_dir) const {
    if (!has_index_) {
        throw std::runtime_error("Cannot save an empty Chimera index");
    }

    prepare_output_directory(index_dir);
    const std::filesystem::path output_dir(index_dir);
    const auto doclens_path = (output_dir / "doclens.bin").string();
    const auto doc_one_bit_path = (output_dir / "doc_1bit.bin").string();
    const auto doc_full_path = (output_dir / "doc_full.bin").string();
    const auto cluster_one_bit_path =
        (output_dir / "cluster_1bit.bin").string();
    const auto ivf_path = (output_dir / "ivf.bin").string();
    const auto centroids_path = (output_dir / "centroids.carga").string();

    write_doc_lens(doc_lens_, doclens_path);
    ivf->save(ivf_path, centroids_path);

    const IndexHeader header {
        n,
        d,
        this->n_clusters,
        this->ex_bits,
        PADDED_DIM,
    };
    auto one_bit_output = open_binary_output(doc_one_bit_path);
    one_bit_output.write(
        reinterpret_cast<const char*>(&header), sizeof(header));
    rotator_->save(one_bit_output);
    one_bit_output.write(
        reinterpret_cast<const char*>(one_bit_codes_.data()),
        static_cast<std::streamsize>(
            one_bit_codes_.size() * sizeof(uint64_t)));
    one_bit_output.write(
        reinterpret_cast<const char*>(one_bit_factors_.data()),
        static_cast<std::streamsize>(
            one_bit_factors_.size() * sizeof(float)));
    one_bit_output.write(
        reinterpret_cast<const char*>(full_bit_factors_.data()),
        static_cast<std::streamsize>(
            full_bit_factors_.size() * sizeof(float)));
    finish_output(one_bit_output, doc_one_bit_path);

    auto full_code_output = open_binary_output(doc_full_path);
    full_code_output.write(
        full_bit_codes_.data(),
        static_cast<std::streamsize>(full_bit_codes_.size()));
    finish_output(full_code_output, doc_full_path);

    auto clustered_output = open_binary_output(cluster_one_bit_path);
    clustered_output.write(
        clustered_data_.data(),
        static_cast<std::streamsize>(clustered_data_.size()));
    finish_output(clustered_output, cluster_one_bit_path);
}

}  // namespace Chimera
