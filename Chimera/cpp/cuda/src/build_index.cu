#include "build_index.hpp"
#include "build_index/assignment.cuh"
#include "build_index/build_utils.cuh"
#include "build_index/centroid_sampling.cuh"
#include "build_index/gpu_quantization.cuh"

#define EIGEN_NO_CUDA

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "gpu_index_layout.hpp"
#include "clustered_format.hpp"
#include "doc_format.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "gpu_config.cuh"
#include "ivf_pg.hpp"
#include "quantization.hpp"

namespace Chimera {

using namespace rabitqlib;

namespace {

constexpr std::uint64_t kDefaultRotatorSeed = 42;
constexpr int kCentroidHnswM = 16;
constexpr int kCentroidHnswEfConstruction = 500;

struct BuildOutputSelection
{
    bool write_centroids_hnsw = true;
    bool write_metadata = true;
};

BuildOutputSelection output_selection_for_mode(BuildOutputMode mode)
{
    BuildOutputSelection selection;
    if (mode == BuildOutputMode::GpuSearchMinimal)
    {
        selection.write_centroids_hnsw = false;
        selection.write_metadata = false;
    }
    return selection;
}

size_t assignment_batch_size()
{
    constexpr size_t kDefaultAssignBatch = 65536;
    const char* value = std::getenv("CHIMERA_BUILD_ASSIGN_BATCH_SIZE");
    if (value == nullptr || *value == '\0')
    {
        return kDefaultAssignBatch;
    }

    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0)
    {
        throw std::runtime_error(
            "CHIMERA_BUILD_ASSIGN_BATCH_SIZE must be a positive integer");
    }
    return static_cast<size_t>(parsed);
}

bool exit_after_step2()
{
    const char* value = std::getenv("CHIMERA_BUILD_EXIT_AFTER_STEP2");
    if (value == nullptr || *value == '\0')
    {
        return false;
    }
    return std::string(value) != "0";
}

struct QuantizedSectionLayout
{
    size_t prefix_bytes = 0;
    size_t one_bit_offset = 0;
    size_t full_code_offset = 0;
    size_t one_bit_factor_offset = 0;
    size_t ex_factor_offset = 0;
    size_t one_bit_bytes_per_vector = 0;
    size_t full_code_bytes_per_vector = 0;
    size_t total_bytes = 0;
};

struct ClusteredStage1SectionLayout
{
    size_t prefix_bytes = 0;
    size_t code_offset = 0;
    size_t factor_offset = 0;
    size_t doc_id_offset = 0;
    size_t token_to_cluster_pos_offset = 0;
    size_t code_bytes_per_vector = 0;
    size_t total_bytes = 0;
};

void write_all_at(
    int fd,
    const void* src,
    size_t bytes,
    size_t offset,
    const std::string& filename)
{
    const auto* ptr = static_cast<const char*>(src);
    size_t written_total = 0;
    while (written_total < bytes)
    {
        const auto cur_offset = static_cast<off_t>(offset + written_total);
        const auto cur_bytes = bytes - written_total;
        const ssize_t written = pwrite(fd, ptr + written_total, cur_bytes, cur_offset);
        if (written < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            throw std::runtime_error(
                "Failed to write quantized batch to " + filename + ": " +
                std::system_category().message(errno));
        }
        written_total += static_cast<size_t>(written);
    }
}

class QuantizedOutputWriter
{
   public:
    QuantizedOutputWriter(
        const std::string& path,
        const doc_format::Header& header,
        const Rotator<float>& rotator,
        size_t n,
        size_t ex_bits)
        : path_(path)
    {
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        if (!output.is_open())
        {
            throw std::runtime_error("Failed to open quantized output file: " + path_);
        }

        doc_format::write_header(output, header, path_);
        doc_format::save_rotator(output, rotator, path_);

        const auto prefix_pos = output.tellp();
        if (prefix_pos == std::streampos(-1))
        {
            throw std::runtime_error("Failed to determine quantized header size for: " + path_);
        }

        layout_.prefix_bytes = static_cast<size_t>(prefix_pos);
        layout_.one_bit_bytes_per_vector = PADDED_DIM / 8;
        layout_.full_code_bytes_per_vector = PADDED_DIM * (1 + ex_bits) / 8;
        layout_.one_bit_offset = layout_.prefix_bytes;
        layout_.full_code_offset =
            layout_.one_bit_offset + n * layout_.one_bit_bytes_per_vector;
        layout_.one_bit_factor_offset =
            layout_.full_code_offset + n * layout_.full_code_bytes_per_vector;
        layout_.ex_factor_offset =
            layout_.one_bit_factor_offset + n * sizeof(float);
        layout_.total_bytes =
            layout_.ex_factor_offset + n * sizeof(float);

        output.close();

        fd_ = open(path_.c_str(), O_WRONLY);
        if (fd_ < 0)
        {
            throw std::runtime_error(
                "Failed to reopen quantized output file: " + path_);
        }
        if (ftruncate(fd_, static_cast<off_t>(layout_.total_bytes)) != 0)
        {
            const auto err = errno;
            close(fd_);
            fd_ = -1;
            throw std::runtime_error(
                "Failed to size quantized output file " + path_ + ": " +
                std::system_category().message(err));
        }
    }

    QuantizedOutputWriter(const QuantizedOutputWriter&) = delete;
    QuantizedOutputWriter& operator=(const QuantizedOutputWriter&) = delete;

    ~QuantizedOutputWriter()
    {
        if (fd_ >= 0)
        {
            close(fd_);
        }
    }

    void write_batch(
        size_t start,
        size_t cur_batch,
        const uint64_t* one_bit_code,
        const uint8_t* full_code,
        const float* one_bit_factor,
        const float* ex_factor)
    {
        const size_t one_bit_bytes = cur_batch * layout_.one_bit_bytes_per_vector;
        const size_t full_code_bytes = cur_batch * layout_.full_code_bytes_per_vector;
        const size_t factor_bytes = cur_batch * sizeof(float);

        write_all_at(
            fd_,
            one_bit_code,
            one_bit_bytes,
            layout_.one_bit_offset + start * layout_.one_bit_bytes_per_vector,
            path_);
        write_all_at(
            fd_,
            full_code,
            full_code_bytes,
            layout_.full_code_offset + start * layout_.full_code_bytes_per_vector,
            path_);
        write_all_at(
            fd_,
            one_bit_factor,
            factor_bytes,
            layout_.one_bit_factor_offset + start * sizeof(float),
            path_);
        write_all_at(
            fd_,
            ex_factor,
            factor_bytes,
            layout_.ex_factor_offset + start * sizeof(float),
            path_);
    }

   private:
    std::string path_;
    int fd_ = -1;
    QuantizedSectionLayout layout_;
};

class QuantizedOutputBuffer
{
   public:
    explicit QuantizedOutputBuffer(size_t n, size_t ex_bits)
        : one_bit_code_(n * PADDED_DIM / 64),
          full_code_(n * PADDED_DIM * (1 + ex_bits) / 8),
          one_bit_factor_(n),
          ex_factor_(n),
          one_bit_bytes_per_vector_(PADDED_DIM / 8),
          full_code_bytes_per_vector_(PADDED_DIM * (1 + ex_bits) / 8)
    {
    }

    void copy_batch(
        size_t start,
        size_t cur_batch,
        const uint64_t* one_bit_code,
        const uint8_t* full_code,
        const float* one_bit_factor,
        const float* ex_factor)
    {
        std::memcpy(
            one_bit_code_.data() + start * (PADDED_DIM / 64),
            one_bit_code,
            cur_batch * one_bit_bytes_per_vector_);
        std::memcpy(
            full_code_.data() + start * full_code_bytes_per_vector_,
            full_code,
            cur_batch * full_code_bytes_per_vector_);
        std::memcpy(
            one_bit_factor_.data() + start,
            one_bit_factor,
            cur_batch * sizeof(float));
        std::memcpy(
            ex_factor_.data() + start,
            ex_factor,
            cur_batch * sizeof(float));
    }

    void write_to_files(
        const std::string& doc_1bit_path,
        const std::string& doc_4bit_path,
        const doc_format::Header& header,
        const Rotator<float>& rotator) const
    {
        std::ofstream doc_1bit_output(doc_1bit_path, std::ios::binary | std::ios::trunc);
        if (!doc_1bit_output.is_open())
        {
            throw std::runtime_error("Failed to open quantized output file: " + doc_1bit_path);
        }
        std::ofstream doc_4bit_output(doc_4bit_path, std::ios::binary | std::ios::trunc);
        if (!doc_4bit_output.is_open())
        {
            throw std::runtime_error("Failed to open quantized output file: " + doc_4bit_path);
        }

        doc_format::write_header(doc_1bit_output, header, doc_1bit_path);
        doc_format::save_rotator(doc_1bit_output, rotator, doc_1bit_path);
        doc_format::write_header(doc_4bit_output, header, doc_4bit_path);
        doc_format::save_rotator(doc_4bit_output, rotator, doc_4bit_path);

        doc_1bit_output.write(
            reinterpret_cast<const char*>(one_bit_code_.data()),
            static_cast<std::streamsize>(one_bit_code_.size() * sizeof(uint64_t)));
        doc_1bit_output.write(
            reinterpret_cast<const char*>(one_bit_factor_.data()),
            static_cast<std::streamsize>(one_bit_factor_.size() * sizeof(float)));
        doc_1bit_output.write(
            reinterpret_cast<const char*>(ex_factor_.data()),
            static_cast<std::streamsize>(ex_factor_.size() * sizeof(float)));

        doc_4bit_output.write(
            reinterpret_cast<const char*>(full_code_.data()),
            static_cast<std::streamsize>(full_code_.size()));
        if (!doc_1bit_output)
        {
            throw std::runtime_error("Failed to write quantized output file: " + doc_1bit_path);
        }
        if (!doc_4bit_output)
        {
            throw std::runtime_error("Failed to write quantized output file: " + doc_4bit_path);
        }
    }

    const uint64_t* one_bit_code_data(size_t start = 0) const
    {
        return one_bit_code_.data() + start * (PADDED_DIM / 64);
    }

    const float* one_bit_factor_data(size_t start = 0) const
    {
        return one_bit_factor_.data() + start;
    }

   private:
    std::vector<uint64_t> one_bit_code_;
    std::vector<uint8_t> full_code_;
    std::vector<float> one_bit_factor_;
    std::vector<float> ex_factor_;
    size_t one_bit_bytes_per_vector_ = 0;
    size_t full_code_bytes_per_vector_ = 0;
};

struct ExSidecarLayout
{
    size_t prefix_bytes = 0;
    size_t ex_code_offset = 0;
    size_t ex_factor_offset = 0;
    size_t ex_code_bytes_per_vector = 0;
    size_t total_bytes = 0;
};

class ExSidecarWriter
{
   public:
    ExSidecarWriter(
        const std::string& path,
        const doc_format::Header& header,
        const Rotator<float>& rotator)
        : path_(path)
    {
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        if (!output.is_open())
        {
            throw std::runtime_error("Failed to open output file: " + path_);
        }

        doc_format::write_header(output, header, path_);
        doc_format::save_rotator(output, rotator, path_);

        const auto prefix_pos = output.tellp();
        if (prefix_pos == std::streampos(-1))
        {
            throw std::runtime_error("Failed to determine sidecar header size for: " + path_);
        }

        layout_.prefix_bytes = static_cast<size_t>(prefix_pos);
        layout_.ex_code_bytes_per_vector =
            doc_format::legacy_ex_code_bytes_per_vector(header);
        layout_.ex_code_offset = layout_.prefix_bytes;
        layout_.ex_factor_offset =
            layout_.ex_code_offset + header.n * layout_.ex_code_bytes_per_vector;
        layout_.total_bytes = layout_.ex_factor_offset + header.n * sizeof(float);
        output.close();

        fd_ = open(path_.c_str(), O_WRONLY);
        if (fd_ < 0)
        {
            throw std::runtime_error("Failed to reopen output file: " + path_);
        }
        if (ftruncate(fd_, static_cast<off_t>(layout_.total_bytes)) != 0)
        {
            const auto err = errno;
            close(fd_);
            fd_ = -1;
            throw std::runtime_error(
                "Failed to size output file " + path_ + ": " +
                std::system_category().message(err));
        }
    }

    ExSidecarWriter(const ExSidecarWriter&) = delete;
    ExSidecarWriter& operator=(const ExSidecarWriter&) = delete;

    ~ExSidecarWriter()
    {
        if (fd_ >= 0)
        {
            close(fd_);
        }
    }

    void write_batch(
        size_t start,
        size_t count,
        const uint8_t* ex_code,
        const float* ex_factor)
    {
        const size_t ex_code_bytes = count * layout_.ex_code_bytes_per_vector;
        const size_t factor_bytes = count * sizeof(float);
        write_all_at(
            fd_,
            ex_code,
            ex_code_bytes,
            layout_.ex_code_offset + start * layout_.ex_code_bytes_per_vector,
            path_);
        write_all_at(
            fd_,
            ex_factor,
            factor_bytes,
            layout_.ex_factor_offset + start * sizeof(float),
            path_);
    }

   private:
    std::string path_;
    int fd_ = -1;
    ExSidecarLayout layout_;
};

struct BatchQuantizationTiming
{
    double rotate_ms = 0.0;
    double h2d_ms = 0.0;
    double encode_ms = 0.0;
    double d2h_ms = 0.0;
    double copy_ms = 0.0;
    double sidecar_ms = 0.0;
    size_t batches = 0;
    size_t embeddings = 0;
};

template <typename T>
class PinnedHostBuffer
{
   public:
    PinnedHostBuffer() = default;

    explicit PinnedHostBuffer(size_t count)
    {
        reset(count);
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~PinnedHostBuffer()
    {
        release();
    }

    void reset(size_t count)
    {
        release();
        if (count == 0)
        {
            return;
        }
        BUILD_CUDA_CHECK(cudaMallocHost(
            reinterpret_cast<void**>(&data_),
            count * sizeof(T)));
        size_ = count;
    }

    T* data()
    {
        return data_;
    }

    const T* data() const
    {
        return data_;
    }

    size_t size() const
    {
        return size_;
    }

   private:
    void release() noexcept
    {
        if (data_ != nullptr)
        {
            (void)cudaFreeHost(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    T* data_ = nullptr;
    size_t size_ = 0;
};

class AssignmentBatchGpuQuantizer
{
   public:
    AssignmentBatchGpuQuantizer(
        int device_id,
        size_t batch_capacity,
        size_t dim,
        size_t full_ex_bits,
        size_t sidecar_ex_bits,
        const Rotator<float>& rotator,
        QuantizedOutputBuffer& output_buffer,
        ExSidecarWriter* sidecar_writer)
        : device_id_(device_id),
          batch_capacity_(batch_capacity),
          dim_(dim),
          full_ex_bits_(full_ex_bits),
          sidecar_ex_bits_(sidecar_ex_bits),
          output_buffer_(output_buffer),
          sidecar_writer_(sidecar_writer),
          one_bit_code_(batch_capacity * PADDED_DIM / 64),
          full_code_(batch_capacity * PADDED_DIM * (1 + full_ex_bits) / 8),
          one_bit_factor_(batch_capacity),
          full_factor_(batch_capacity),
          sidecar_code_(
              sidecar_ex_bits == 0 ?
              0 :
              batch_capacity * PADDED_DIM * sidecar_ex_bits / 8),
          sidecar_factor_(sidecar_ex_bits == 0 ? 0 : batch_capacity)
    {
        const auto* fht_rotator =
            dynamic_cast<const rotator_impl::FhtKacRotator*>(&rotator);
        if (fht_rotator == nullptr || dim_ != PADDED_DIM)
        {
            throw std::runtime_error(
                "GPU rotation currently requires FhtKacRotator with dim=128");
        }
        BUILD_CUDA_CHECK(cudaSetDevice(device_id_));
        BUILD_CUDA_CHECK(cudaStreamCreate(&stream_));
        BUILD_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_raw_),
            batch_capacity_ * dim_ * sizeof(float)));
        BUILD_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_fht_flip_),
            fht_rotator->flip_bytes()));
        BUILD_CUDA_CHECK(cudaMemcpyAsync(
            d_fht_flip_,
            fht_rotator->flip_data(),
            fht_rotator->flip_bytes(),
            cudaMemcpyHostToDevice,
            stream_));
        BUILD_CUDA_CHECK(cudaStreamSynchronize(stream_));
        fht_flip_bytes_ = fht_rotator->flip_bytes();
        BUILD_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_one_bit_code_),
            one_bit_code_.size() * sizeof(uint64_t)));
        BUILD_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_full_code_),
            full_code_.size() * sizeof(uint8_t)));
        BUILD_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_one_bit_factor_),
            one_bit_factor_.size() * sizeof(float)));
        BUILD_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_full_factor_),
            full_factor_.size() * sizeof(float)));
        if (sidecar_ex_bits_ != 0)
        {
            BUILD_CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_sidecar_code_),
                sidecar_code_.size() * sizeof(uint8_t)));
            BUILD_CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_sidecar_factor_),
                sidecar_factor_.size() * sizeof(float)));
        }
    }

    AssignmentBatchGpuQuantizer(const AssignmentBatchGpuQuantizer&) = delete;
    AssignmentBatchGpuQuantizer& operator=(const AssignmentBatchGpuQuantizer&) = delete;

    ~AssignmentBatchGpuQuantizer()
    {
        if (d_sidecar_factor_ != nullptr) (void)cudaFree(d_sidecar_factor_);
        if (d_sidecar_code_ != nullptr) (void)cudaFree(d_sidecar_code_);
        if (d_full_factor_ != nullptr) (void)cudaFree(d_full_factor_);
        if (d_one_bit_factor_ != nullptr) (void)cudaFree(d_one_bit_factor_);
        if (d_full_code_ != nullptr) (void)cudaFree(d_full_code_);
        if (d_one_bit_code_ != nullptr) (void)cudaFree(d_one_bit_code_);
        if (d_fht_flip_ != nullptr) (void)cudaFree(d_fht_flip_);
        if (d_raw_ != nullptr) (void)cudaFree(d_raw_);
        if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    }

    void quantize(const AssignmentBatch& batch)
    {
        if (batch.count > batch_capacity_)
        {
            throw std::runtime_error("assignment quantization batch exceeds capacity");
        }
        if (batch.dim != dim_)
        {
            throw std::runtime_error("assignment quantization batch dimension mismatch");
        }
        BUILD_CUDA_CHECK(cudaSetDevice(device_id_));

        const size_t raw_bytes = batch.count * dim_ * sizeof(float);
        const size_t one_bit_code_bytes =
            batch.count * (PADDED_DIM / 64) * sizeof(uint64_t);
        const size_t full_code_bytes =
            batch.count * PADDED_DIM * (1 + full_ex_bits_) / 8;
        const size_t sidecar_code_bytes =
            sidecar_ex_bits_ == 0 ? 0 : batch.count * PADDED_DIM * sidecar_ex_bits_ / 8;

        const auto h2d_start = std::chrono::steady_clock::now();
        BUILD_CUDA_CHECK(cudaMemcpyAsync(
            d_raw_,
            batch.embeddings,
            raw_bytes,
            cudaMemcpyHostToDevice,
            stream_));
        BUILD_CUDA_CHECK(cudaStreamSynchronize(stream_));
        timing_.h2d_ms += elapsed_ms(h2d_start, std::chrono::steady_clock::now());

        GpuRotateQuantizeBatchParams params;
        params.d_raw = d_raw_;
        params.count = batch.count;
        params.dim = dim_;
        params.d_fht_flip = d_fht_flip_;
        params.fht_flip_bytes = fht_flip_bytes_;
        params.d_one_bit_code = d_one_bit_code_;
        params.d_one_bit_factor = d_one_bit_factor_;
        params.full_ex_bits = full_ex_bits_;
        params.d_full_code = d_full_code_;
        params.d_full_factor = d_full_factor_;
        params.sidecar_ex_bits = sidecar_ex_bits_;
        params.d_sidecar_code = d_sidecar_code_;
        params.d_sidecar_factor = d_sidecar_factor_;
        params.stream = stream_;

        const auto encode_start = std::chrono::steady_clock::now();
        gpu_rotate_quantize_batch(params);
        BUILD_CUDA_CHECK(cudaStreamSynchronize(stream_));
        timing_.encode_ms += elapsed_ms(encode_start, std::chrono::steady_clock::now());

        const auto d2h_start = std::chrono::steady_clock::now();
        BUILD_CUDA_CHECK(cudaMemcpyAsync(
            one_bit_code_.data(),
            d_one_bit_code_,
            one_bit_code_bytes,
            cudaMemcpyDeviceToHost,
            stream_));
        BUILD_CUDA_CHECK(cudaMemcpyAsync(
            full_code_.data(),
            d_full_code_,
            full_code_bytes,
            cudaMemcpyDeviceToHost,
            stream_));
        BUILD_CUDA_CHECK(cudaMemcpyAsync(
            one_bit_factor_.data(),
            d_one_bit_factor_,
            batch.count * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_));
        BUILD_CUDA_CHECK(cudaMemcpyAsync(
            full_factor_.data(),
            d_full_factor_,
            batch.count * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_));
        if (sidecar_ex_bits_ != 0)
        {
            BUILD_CUDA_CHECK(cudaMemcpyAsync(
                sidecar_code_.data(),
                d_sidecar_code_,
                sidecar_code_bytes,
                cudaMemcpyDeviceToHost,
                stream_));
            BUILD_CUDA_CHECK(cudaMemcpyAsync(
                sidecar_factor_.data(),
                d_sidecar_factor_,
                batch.count * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream_));
        }
        BUILD_CUDA_CHECK(cudaStreamSynchronize(stream_));
        timing_.d2h_ms += elapsed_ms(d2h_start, std::chrono::steady_clock::now());

        const auto copy_start = std::chrono::steady_clock::now();
        output_buffer_.copy_batch(
            batch.start,
            batch.count,
            one_bit_code_.data(),
            full_code_.data(),
            one_bit_factor_.data(),
            full_factor_.data());
        timing_.copy_ms += elapsed_ms(copy_start, std::chrono::steady_clock::now());

        if (sidecar_writer_ != nullptr)
        {
            const auto sidecar_start = std::chrono::steady_clock::now();
            sidecar_writer_->write_batch(
                batch.start,
                batch.count,
                sidecar_code_.data(),
                sidecar_factor_.data());
            timing_.sidecar_ms += elapsed_ms(sidecar_start, std::chrono::steady_clock::now());
        }

        timing_.batches += 1;
        timing_.embeddings += batch.count;
    }

    const BatchQuantizationTiming& timing() const
    {
        return timing_;
    }

   private:
    int device_id_ = 0;
    size_t batch_capacity_ = 0;
    size_t dim_ = 0;
    size_t full_ex_bits_ = 0;
    size_t sidecar_ex_bits_ = 0;
    QuantizedOutputBuffer& output_buffer_;
    ExSidecarWriter* sidecar_writer_ = nullptr;
    cudaStream_t stream_ = nullptr;
    PinnedHostBuffer<uint64_t> one_bit_code_;
    PinnedHostBuffer<uint8_t> full_code_;
    PinnedHostBuffer<float> one_bit_factor_;
    PinnedHostBuffer<float> full_factor_;
    PinnedHostBuffer<uint8_t> sidecar_code_;
    PinnedHostBuffer<float> sidecar_factor_;
    float* d_raw_ = nullptr;
    uint8_t* d_fht_flip_ = nullptr;
    size_t fht_flip_bytes_ = 0;
    uint64_t* d_one_bit_code_ = nullptr;
    uint8_t* d_full_code_ = nullptr;
    float* d_one_bit_factor_ = nullptr;
    float* d_full_factor_ = nullptr;
    uint8_t* d_sidecar_code_ = nullptr;
    float* d_sidecar_factor_ = nullptr;
    BatchQuantizationTiming timing_;
};

void build_centroid_hnsw_from_host_data(
    const std::vector<float>& centroids,
    size_t n_centroids,
    size_t dim,
    const std::string& output_path)
{
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> hnsw(&space, n_centroids, kCentroidHnswM, kCentroidHnswEfConstruction);
#pragma omp parallel for
    for (size_t i = 0; i < n_centroids; ++i) {
        hnsw.addPoint(centroids.data() + i * dim, i);
    }
    hnsw.saveIndex(output_path);
}

class ClusteredStage1OutputWriter
{
   public:
    ClusteredStage1OutputWriter(
        const std::string& path,
        size_t n_entries,
        size_t code_bytes_per_vector)
        : path_(path)
    {
        clustered_format::Header header;
        header.version = 2;
        header.n_entries = n_entries;
        header.code_bytes_per_vector = code_bytes_per_vector;

        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        if (!output.is_open())
        {
            throw std::runtime_error("Failed to open clustered sidecar output file: " + path_);
        }

        clustered_format::write_header(output, header, path_);

        const auto prefix_pos = output.tellp();
        if (prefix_pos == std::streampos(-1))
        {
            throw std::runtime_error("Failed to determine clustered sidecar header size for: " + path_);
        }

        layout_.prefix_bytes = static_cast<size_t>(prefix_pos);
        layout_.code_bytes_per_vector = code_bytes_per_vector;
        layout_.code_offset = layout_.prefix_bytes;
        layout_.factor_offset = layout_.code_offset + n_entries * code_bytes_per_vector;
        layout_.doc_id_offset = layout_.factor_offset + n_entries * sizeof(float);
        layout_.token_to_cluster_pos_offset = layout_.doc_id_offset + n_entries * sizeof(int);
        layout_.total_bytes =
            layout_.token_to_cluster_pos_offset + n_entries * sizeof(uint32_t);
        output.close();

        fd_ = open(path_.c_str(), O_RDWR);
        if (fd_ < 0)
        {
            throw std::runtime_error(
                "Failed to reopen clustered sidecar output file: " + path_);
        }
        if (ftruncate(fd_, static_cast<off_t>(layout_.total_bytes)) != 0)
        {
            const auto err = errno;
            close(fd_);
            fd_ = -1;
            throw std::runtime_error(
                "Failed to size clustered sidecar output file " + path_ + ": " +
                std::system_category().message(err));
        }

        mapping_ = mmap(nullptr, layout_.total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapping_ == MAP_FAILED)
        {
            const auto err = errno;
            close(fd_);
            fd_ = -1;
            mapping_ = nullptr;
            throw std::runtime_error(
                "Failed to mmap clustered sidecar output file " + path_ + ": " +
                std::system_category().message(err));
        }

        code_data_ = static_cast<char*>(mapping_) + layout_.code_offset;
        factor_data_ = reinterpret_cast<float*>(static_cast<char*>(mapping_) + layout_.factor_offset);
        doc_id_data_ = reinterpret_cast<int*>(static_cast<char*>(mapping_) + layout_.doc_id_offset);
        token_to_cluster_pos_data_ = reinterpret_cast<uint32_t*>(
            static_cast<char*>(mapping_) + layout_.token_to_cluster_pos_offset);
        if (n_entries > 0)
        {
            current_doc_end_ = 0;
        }
    }

    ClusteredStage1OutputWriter(const ClusteredStage1OutputWriter&) = delete;
    ClusteredStage1OutputWriter& operator=(const ClusteredStage1OutputWriter&) = delete;

    ~ClusteredStage1OutputWriter()
    {
        if (mapping_ != nullptr)
        {
            msync(mapping_, layout_.total_bytes, MS_SYNC);
            munmap(mapping_, layout_.total_bytes);
        }
        if (fd_ >= 0)
        {
            close(fd_);
        }
    }

    void scatter_batch(
        size_t start,
        size_t cur_batch,
        const uint64_t* one_bit_code,
        const float* one_bit_factor,
        const uint32_t* cluster_rank,
        const std::vector<int>& doc_ptrs)
    {
        const size_t code_bytes = layout_.code_bytes_per_vector;
        size_t doc_id = current_doc_id_;
        size_t doc_end = current_doc_end_;
        if (doc_ptrs.size() > 1 && doc_end == 0)
        {
            doc_end = static_cast<size_t>(doc_ptrs[1]);
        }

        for (size_t i = 0; i < cur_batch; ++i)
        {
            const size_t orig_id = start + i;
            while (orig_id >= doc_end && doc_id + 1 < doc_ptrs.size() - 1)
            {
                ++doc_id;
                doc_end = static_cast<size_t>(doc_ptrs[doc_id + 1]);
            }

            const size_t clustered_pos = cluster_rank[orig_id];
            std::memcpy(
                code_data_ + clustered_pos * code_bytes,
                reinterpret_cast<const char*>(one_bit_code) + i * code_bytes,
                code_bytes);
            factor_data_[clustered_pos] = one_bit_factor[i];
            doc_id_data_[clustered_pos] = static_cast<int>(doc_id);
            token_to_cluster_pos_data_[orig_id] = static_cast<uint32_t>(clustered_pos);
        }

        current_doc_id_ = doc_id;
        current_doc_end_ = doc_end;
    }

   private:
    std::string path_;
    int fd_ = -1;
    void* mapping_ = nullptr;
    ClusteredStage1SectionLayout layout_;
    char* code_data_ = nullptr;
    float* factor_data_ = nullptr;
    int* doc_id_data_ = nullptr;
    uint32_t* token_to_cluster_pos_data_ = nullptr;
    size_t current_doc_id_ = 0;
    size_t current_doc_end_ = 0;
};

class ClusteredStage1Buffer
{
   public:
    ClusteredStage1Buffer(size_t n_entries, size_t code_bytes_per_vector)
        : code_(n_entries * code_bytes_per_vector),
          factor_(n_entries),
          doc_id_(n_entries),
          token_to_cluster_pos_(n_entries),
          code_bytes_per_vector_(code_bytes_per_vector)
    {
    }

    void scatter_batch(
        size_t start,
        size_t cur_batch,
        const uint64_t* one_bit_code,
        const float* one_bit_factor,
        const uint32_t* cluster_rank,
        const std::vector<int>& doc_ptrs)
    {
        size_t doc_id = current_doc_id_;
        size_t doc_end = current_doc_end_;
        if (doc_ptrs.size() > 1 && doc_end == 0)
        {
            doc_end = static_cast<size_t>(doc_ptrs[1]);
        }

        for (size_t i = 0; i < cur_batch; ++i)
        {
            const size_t orig_id = start + i;
            while (orig_id >= doc_end && doc_id + 1 < doc_ptrs.size() - 1)
            {
                ++doc_id;
                doc_end = static_cast<size_t>(doc_ptrs[doc_id + 1]);
            }

            const size_t clustered_pos = cluster_rank[orig_id];
            std::memcpy(
                code_.data() + clustered_pos * code_bytes_per_vector_,
                reinterpret_cast<const char*>(one_bit_code) + i * code_bytes_per_vector_,
                code_bytes_per_vector_);
            factor_[clustered_pos] = one_bit_factor[i];
            doc_id_[clustered_pos] = static_cast<int>(doc_id);
            token_to_cluster_pos_[orig_id] = static_cast<uint32_t>(clustered_pos);
        }

        current_doc_id_ = doc_id;
        current_doc_end_ = doc_end;
    }

    void write_to_file(const std::string& path) const
    {
        clustered_format::Header header;
        header.version = 2;
        header.n_entries = factor_.size();
        header.code_bytes_per_vector = code_bytes_per_vector_;

        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output.is_open())
        {
            throw std::runtime_error("Failed to open clustered stage-1 output file: " + path);
        }

        clustered_format::write_header(output, header, path);
        output.write(
            code_.data(),
            static_cast<std::streamsize>(code_.size()));
        output.write(
            reinterpret_cast<const char*>(factor_.data()),
            static_cast<std::streamsize>(factor_.size() * sizeof(float)));
        output.write(
            reinterpret_cast<const char*>(doc_id_.data()),
            static_cast<std::streamsize>(doc_id_.size() * sizeof(int)));
        output.write(
            reinterpret_cast<const char*>(token_to_cluster_pos_.data()),
            static_cast<std::streamsize>(token_to_cluster_pos_.size() * sizeof(uint32_t)));
        if (!output)
        {
            throw std::runtime_error("Failed to write clustered stage-1 output file: " + path);
        }
    }

   private:
    std::vector<char> code_;
    std::vector<float> factor_;
    std::vector<int> doc_id_;
    std::vector<uint32_t> token_to_cluster_pos_;
    size_t code_bytes_per_vector_ = 0;
    size_t current_doc_id_ = 0;
    size_t current_doc_end_ = 0;
};

void write_index_metadata_json(
    const std::string& index_dir,
    const doc_format::Header& header,
    int max_cluster_size_tokens,
    const CentroidSampleOptions& centroid_sample_options,
    QuantizationMode quantization_mode,
    size_t sidecar_ex_bits,
    QuantizationBackend quantization_backend)
{
    clustered_format::Header cluster_header;
    cluster_header.version = 2;
    cluster_header.n_entries = header.n;
    cluster_header.code_bytes_per_vector =
        doc_format::one_bit_bytes_per_vector(header);

    auto ex_sidecar_header = header;
    ex_sidecar_header.ex_bits = sidecar_ex_bits;

    const auto path = gpu_index_layout::metadata_path(index_dir);
    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open())
    {
        throw std::runtime_error("Failed to open metadata output file: " + path);
    }

    const auto doclens_path = gpu_index_layout::doclens_path(index_dir);
    const auto doc_1bit_path = gpu_index_layout::doc_1bit_path(index_dir);
    const auto doc_4bit_path = gpu_index_layout::doc_4bit_path(index_dir);
    const auto doc_4bit_ex_path = gpu_index_layout::doc_4bit_ex_path(index_dir);
    const auto cluster_1bit_path = gpu_index_layout::cluster_1bit_path(index_dir);
    const auto ivf_path = gpu_index_layout::ivf_path(index_dir);
    const auto centroids_path = gpu_index_layout::centroids_path(index_dir);
    const auto centroids_hnsw_path = gpu_index_layout::centroids_hnsw_path(index_dir);

    const auto doc_prefix_bytes = doc_format::prefix_bytes(header);
    const auto doc_4bit_ex_prefix_bytes = doc_format::prefix_bytes(ex_sidecar_header);
    const auto cluster_prefix_bytes = clustered_format::prefix_bytes(cluster_header);
    const auto one_bit_code_bytes = doc_format::one_bit_code_bytes(header);
    const auto one_bit_factor_bytes = doc_format::one_bit_factor_bytes(header);
    const auto ex_factor_bytes = doc_format::ex_factor_bytes(header);
    const auto full_code_bytes = doc_format::full_code_bytes(header);
    const auto legacy_ex_code_bytes = doc_format::legacy_ex_code_bytes(ex_sidecar_header);
    const auto cluster_code_bytes = header.n * cluster_header.code_bytes_per_vector;
    const auto cluster_factor_bytes = header.n * sizeof(float);
    const auto cluster_doc_id_bytes = header.n * sizeof(int);
    const auto cluster_token_to_cluster_pos_bytes = header.n * sizeof(uint32_t);

    output
        << "{\n"
        << "  \"format\": \"chimera_split_index_v2\",\n"
        << "  \"build_status\": \"complete\",\n"
        << "  \"builder\": \"gpu_build\",\n"
        << "  \"built_at_utc\": \"" << current_utc_timestamp() << "\",\n"
        << "  \"centroid_sample_seed\": " << centroid_sample_options.seed << ",\n"
        << "  \"centroid_sample_bucket_size\": "
        << std::max<size_t>(1, centroid_sample_options.bucket_size) << ",\n"
        << "  \"rotator_seed\": " << kDefaultRotatorSeed << ",\n"
        << "  \"n\": " << header.n << ",\n"
        << "  \"d\": " << header.d << ",\n"
        << "  \"n_clusters\": " << header.n_clusters << ",\n"
        << "  \"quantization\": \"" << quantization_mode_name(quantization_mode) << "\",\n"
        << "  \"quantization_backend\": \"" << quantization_backend_name(quantization_backend) << "\",\n"
        << "  \"ex_bits\": " << header.ex_bits << ",\n"
        << "  \"sidecar_ex_bits\": " << sidecar_ex_bits << ",\n"
        << "  \"padded_dim\": " << header.padded_dim << ",\n"
        << "  \"rotator_type\": \"" << doc_format::rotator_type_name(header.rotator_type) << "\",\n"
        << "  \"max_cluster_size_tokens\": " << max_cluster_size_tokens << ",\n"
        << "  \"cluster_token_ordering\": \"original_token_id_ascending_within_cluster (equivalent to doc_id_then_token_id for natural document order)\",\n"
        << "  \"files\": {\n"
        << "    \"doclens\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kDoclensFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(doclens_path) << ",\n"
        << "      \"content\": \"int32 document lengths, one entry per document\"\n"
        << "    },\n"
        << "    \"doc_1bit\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kDoc1BitFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(doc_1bit_path) << ",\n"
        << "      \"prefix_bytes\": " << doc_prefix_bytes << ",\n"
        << "      \"one_bit_code_offset_bytes\": " << doc_prefix_bytes << ",\n"
        << "      \"one_bit_code_bytes\": " << one_bit_code_bytes << ",\n"
        << "      \"one_bit_factor_offset_bytes\": " << (doc_prefix_bytes + one_bit_code_bytes) << ",\n"
        << "      \"one_bit_factor_bytes\": " << one_bit_factor_bytes << ",\n"
        << "      \"ex_factor_offset_bytes\": " << (doc_prefix_bytes + one_bit_code_bytes + one_bit_factor_bytes) << ",\n"
        << "      \"ex_factor_bytes\": " << ex_factor_bytes << ",\n"
        << "      \"content\": \"original-order stage-1 payload with embedded rotator\"\n"
        << "    },\n"
        << "    \"doc_4bit\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kDoc4BitFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(doc_4bit_path) << ",\n"
        << "      \"prefix_bytes\": " << doc_prefix_bytes << ",\n"
        << "      \"full_code_offset_bytes\": " << doc_prefix_bytes << ",\n"
        << "      \"full_code_bytes\": " << full_code_bytes << ",\n"
        << "      \"content\": \"original-order fused 1+ex bit codes for stage-2 GPU reranking\"\n"
        << "    },\n"
        << "    \"doc_4bit_ex\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kDoc4BitExFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(doc_4bit_ex_path) << ",\n"
        << "      \"prefix_bytes\": " << doc_4bit_ex_prefix_bytes << ",\n"
        << "      \"ex_bits\": " << sidecar_ex_bits << ",\n"
        << "      \"ex_code_offset_bytes\": " << doc_4bit_ex_prefix_bytes << ",\n"
        << "      \"ex_code_bytes\": " << legacy_ex_code_bytes << ",\n"
        << "      \"ex_factor_offset_bytes\": " << (doc_4bit_ex_prefix_bytes + legacy_ex_code_bytes) << ",\n"
        << "      \"ex_factor_bytes\": " << doc_format::ex_factor_bytes(ex_sidecar_header) << ",\n"
        << "      \"content\": \"original-order residual ex-bit sidecar for CPU reranking\"\n"
        << "    },\n"
        << "    \"cluster_1bit\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kCluster1BitFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(cluster_1bit_path) << ",\n"
        << "      \"prefix_bytes\": " << cluster_prefix_bytes << ",\n"
        << "      \"one_bit_code_offset_bytes\": " << cluster_prefix_bytes << ",\n"
        << "      \"one_bit_code_bytes\": " << cluster_code_bytes << ",\n"
        << "      \"one_bit_factor_offset_bytes\": " << (cluster_prefix_bytes + cluster_code_bytes) << ",\n"
        << "      \"one_bit_factor_bytes\": " << cluster_factor_bytes << ",\n"
        << "      \"doc_id_offset_bytes\": " << (cluster_prefix_bytes + cluster_code_bytes + cluster_factor_bytes) << ",\n"
        << "      \"doc_id_bytes\": " << cluster_doc_id_bytes << ",\n"
        << "      \"token_to_cluster_pos_offset_bytes\": "
        << (cluster_prefix_bytes + cluster_code_bytes + cluster_factor_bytes + cluster_doc_id_bytes) << ",\n"
        << "      \"token_to_cluster_pos_bytes\": " << cluster_token_to_cluster_pos_bytes << ",\n"
        << "      \"content\": \"cluster-ordered stage-1 sidecar with tokens ordered by original token id inside each cluster\"\n"
        << "    },\n"
        << "    \"ivf\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kIvfFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(ivf_path) << ",\n"
        << "      \"content\": \"IVF posting lists and cluster boundaries\"\n"
        << "    },\n"
        << "    \"centroids\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kCentroidsFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(centroids_path) << ",\n"
        << "      \"content\": \"persisted rotated CAGRA centroid graph\"\n"
        << "    },\n"
        << "    \"centroids_hnsw\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kCentroidsHnswFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(centroids_hnsw_path) << ",\n"
        << "      \"M\": " << kCentroidHnswM << ",\n"
        << "      \"ef_construction\": " << kCentroidHnswEfConstruction << ",\n"
        << "      \"content\": \"hnswlib HNSW index built from the in-memory rotated centroids\"\n"
        << "    }\n"
        << "  }\n"
        << "}\n";
    if (!output)
    {
        throw std::runtime_error("Failed to write metadata output file: " + path);
    }
}

void copy_stream_bytes(
    std::ifstream& input,
    std::ofstream& output,
    size_t bytes,
    const std::string& input_path,
    const std::string& output_path)
{
    constexpr size_t kCopyBufferBytes = 8ULL * 1024ULL * 1024ULL;
    std::vector<char> buffer(std::min(kCopyBufferBytes, std::max<size_t>(bytes, 1)));
    size_t remaining = bytes;
    while (remaining > 0)
    {
        const size_t cur_bytes = std::min(remaining, buffer.size());
        input.read(buffer.data(), static_cast<std::streamsize>(cur_bytes));
        if (!input)
        {
            throw std::runtime_error(
                "Failed to read " + std::to_string(cur_bytes) + " bytes from " + input_path);
        }
        output.write(buffer.data(), static_cast<std::streamsize>(cur_bytes));
        if (!output)
        {
            throw std::runtime_error(
                "Failed to write " + std::to_string(cur_bytes) + " bytes to " + output_path);
        }
        remaining -= cur_bytes;
    }
}

void copy_split_quantized_layout(
    const gpu_index_layout::ResolvedPaths& source_paths,
    const std::string& target_doc_1bit_path,
    const std::string& target_doc_4bit_path,
    const std::string& target_doc_4bit_ex_path,
    const std::string& target_centroids_hnsw_path)
{
    if (source_paths.doc_4bit_path.empty())
    {
        throw std::runtime_error(
            "Missing doc_4bit.bin next to " + source_paths.quantized_data_path);
    }

    if (source_paths.quantized_data_path != target_doc_1bit_path)
    {
        std::filesystem::copy_file(
            source_paths.quantized_data_path,
            target_doc_1bit_path,
            std::filesystem::copy_options::overwrite_existing);
    }
    if (source_paths.doc_4bit_path != target_doc_4bit_path)
    {
        std::filesystem::copy_file(
            source_paths.doc_4bit_path,
            target_doc_4bit_path,
            std::filesystem::copy_options::overwrite_existing);
    }
    if (!source_paths.doc_4bit_ex_path.empty() &&
        std::filesystem::exists(source_paths.doc_4bit_ex_path) &&
        source_paths.doc_4bit_ex_path != target_doc_4bit_ex_path)
    {
        std::filesystem::copy_file(
            source_paths.doc_4bit_ex_path,
            target_doc_4bit_ex_path,
            std::filesystem::copy_options::overwrite_existing);
    }
    if (!source_paths.centroids_hnsw_path.empty() &&
        std::filesystem::exists(source_paths.centroids_hnsw_path) &&
        source_paths.centroids_hnsw_path != target_centroids_hnsw_path)
    {
        std::filesystem::copy_file(
            source_paths.centroids_hnsw_path,
            target_centroids_hnsw_path,
            std::filesystem::copy_options::overwrite_existing);
    }
}

}  // namespace

BuildOutputMode parse_build_output_mode(const std::string& value)
{
    std::string normalized = value;
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    std::replace(normalized.begin(), normalized.end(), '_', '-');

    if (normalized == "full")
    {
        return BuildOutputMode::Full;
    }
    if (normalized == "gpu-search-minimal" || normalized == "minimal" ||
        normalized == "gpu-minimal")
    {
        return BuildOutputMode::GpuSearchMinimal;
    }
    throw std::runtime_error(
        "build mode must be full or gpu-search-minimal, got: " + value);
}

const char* build_output_mode_name(BuildOutputMode mode)
{
    switch (mode)
    {
    case BuildOutputMode::Full:
        return "full";
    case BuildOutputMode::GpuSearchMinimal:
        return "gpu-search-minimal";
    }
    return "unknown";
}

QuantizationMode parse_quantization_mode(const std::string& value)
{
    std::string normalized = value;
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    std::replace(normalized.begin(), normalized.end(), '-', '_');

    if (normalized == "4bit" || normalized == "4")
    {
        return QuantizationMode::Bit4;
    }
    if (normalized == "4bit_ex" || normalized == "4ex")
    {
        return QuantizationMode::Bit4Ex;
    }
    if (normalized == "8bit" || normalized == "8")
    {
        return QuantizationMode::Bit8;
    }
    if (normalized == "8bit_ex" || normalized == "8ex")
    {
        return QuantizationMode::Bit8Ex;
    }
    throw std::runtime_error(
        "quantization must be 4bit, 4bit_ex, 8bit, or 8bit_ex, got: " + value);
}

const char* quantization_mode_name(QuantizationMode mode)
{
    switch (mode)
    {
    case QuantizationMode::Bit4:
        return "4bit";
    case QuantizationMode::Bit4Ex:
        return "4bit_ex";
    case QuantizationMode::Bit8:
        return "8bit";
    case QuantizationMode::Bit8Ex:
        return "8bit_ex";
    }
    return "unknown";
}

size_t quantization_full_ex_bits(QuantizationMode mode)
{
    switch (mode)
    {
    case QuantizationMode::Bit4:
    case QuantizationMode::Bit4Ex:
        return 3;
    case QuantizationMode::Bit8:
    case QuantizationMode::Bit8Ex:
        return 7;
    }
    throw std::runtime_error("unknown quantization mode");
}

size_t quantization_sidecar_ex_bits(QuantizationMode mode)
{
    switch (mode)
    {
    case QuantizationMode::Bit4:
    case QuantizationMode::Bit8:
        return 0;
    case QuantizationMode::Bit4Ex:
        return 4;
    case QuantizationMode::Bit8Ex:
        return 8;
    }
    throw std::runtime_error("unknown quantization mode");
}

QuantizationBackend parse_quantization_backend(const std::string& value)
{
    std::string normalized = value;
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    std::replace(normalized.begin(), normalized.end(), '-', '_');

    if (normalized == "cpu")
    {
        return QuantizationBackend::Cpu;
    }
    if (normalized == "gpu" || normalized == "cuda")
    {
        return QuantizationBackend::Gpu;
    }
    if (normalized == "gpu_merged" || normalized == "merged_gpu" ||
        normalized == "cuda_merged" || normalized == "merged_cuda")
    {
        return QuantizationBackend::GpuMerged;
    }
    throw std::runtime_error(
        "quantization backend must be cpu, gpu, or gpu-merged, got: " + value);
}

const char* quantization_backend_name(QuantizationBackend backend)
{
    switch (backend)
    {
    case QuantizationBackend::Cpu:
        return "cpu";
    case QuantizationBackend::Gpu:
        return "gpu";
    case QuantizationBackend::GpuMerged:
        return "gpu-merged";
    }
    return "unknown";
}

void build_index(
    const MmapedEmbeddings &data,
    size_t n_clusters,
    size_t ex_bits,
    const std::vector<int> &doc_lens,
    const std::string &index_dir,
    const CentroidSampleOptions& centroid_sample_options)
{
    BuildIndexOptions options;
    options.centroid_sample = centroid_sample_options;
    options.output_mode = BuildOutputMode::Full;
    if (ex_bits == 3)
    {
        options.quantization_mode = QuantizationMode::Bit4;
    }
    else if (ex_bits == 7)
    {
        options.quantization_mode = QuantizationMode::Bit8;
    }
    else
    {
        throw std::runtime_error("gpu build supports ex_bits=3 or 7");
    }
    build_index(data, n_clusters, ex_bits, doc_lens, index_dir, options);
}

void build_index(
    const MmapedEmbeddings &data,
    size_t n_clusters,
    size_t ex_bits,
    const std::vector<int> &doc_lens,
    const std::string &index_dir,
    const BuildIndexOptions& options)
{
    using Clock = std::chrono::steady_clock;
    const auto output_selection = output_selection_for_mode(options.output_mode);
    const auto& centroid_sample_options = options.centroid_sample;

    const size_t n = data.num_embeddings;
    const size_t d = data.d;
    const auto build_start = Clock::now();

    std::filesystem::create_directories(index_dir);
    std::error_code cleanup_ec;
    std::filesystem::remove(
        gpu_index_layout::doc_1bit_path(index_dir), cleanup_ec);
    cleanup_ec.clear();
    std::filesystem::remove(
        gpu_index_layout::doc_4bit_path(index_dir), cleanup_ec);
    cleanup_ec.clear();
    std::filesystem::remove(
        gpu_index_layout::cluster_1bit_path(index_dir), cleanup_ec);
    cleanup_ec.clear();
    std::filesystem::remove(
        gpu_index_layout::doc_4bit_ex_path(index_dir), cleanup_ec);
    cleanup_ec.clear();
    std::filesystem::remove(
        gpu_index_layout::centroids_hnsw_path(index_dir), cleanup_ec);
    cleanup_ec.clear();
    std::filesystem::remove(
        gpu_index_layout::metadata_path(index_dir), cleanup_ec);
    cleanup_ec.clear();
    std::filesystem::remove(
        gpu_index_layout::cpu_index_path(index_dir), cleanup_ec);
    cleanup_ec.clear();
    std::filesystem::remove(
        gpu_index_layout::gpu_index_path(index_dir), cleanup_ec);

    const std::string ivf_path = gpu_index_layout::ivf_path(index_dir);
    const std::string doc_1bit_path = gpu_index_layout::doc_1bit_path(index_dir);
    const std::string doc_4bit_path = gpu_index_layout::doc_4bit_path(index_dir);
    const std::string doc_4bit_ex_path = gpu_index_layout::doc_4bit_ex_path(index_dir);
    const std::string centroids_path = gpu_index_layout::centroids_path(index_dir);
    const std::string centroids_hnsw_path = gpu_index_layout::centroids_hnsw_path(index_dir);
    const std::string clustered_stage1_path = gpu_index_layout::cluster_1bit_path(index_dir);
    const std::string metadata_path = gpu_index_layout::metadata_path(index_dir);
    ex_bits = quantization_full_ex_bits(options.quantization_mode);
    const size_t sidecar_ex_bits =
        quantization_sidecar_ex_bits(options.quantization_mode);
    const bool write_ex_sidecar = sidecar_ex_bits != 0;

    std::cout << "[build_index] build_mode="
              << build_output_mode_name(options.output_mode)
              << ", quantization="
              << quantization_mode_name(options.quantization_mode)
              << " (full_ex_bits=" << ex_bits
              << ", sidecar_ex_bits=" << sidecar_ex_bits << ")"
              << ", quantization_backend="
              << quantization_backend_name(options.quantization_backend)
              << ", write_centroids_hnsw="
              << (output_selection.write_centroids_hnsw ? "yes" : "no")
              << ", write_ex_sidecar="
              << (write_ex_sidecar ? "yes" : "no")
              << ", write_metadata="
              << (output_selection.write_metadata ? "yes" : "no")
              << "." << std::endl;

    std::cout << "[build_index] Step 1: Randomly sampling " << n_clusters
              << " centroids from " << n << " vectors (d=" << d
              << ", bucket_size="
              << std::max<size_t>(1, centroid_sample_options.bucket_size)
              << ", seed=" << centroid_sample_options.seed
              << ") ..." << std::endl;
    const auto step1_start = Clock::now();

    std::vector<float> centroids =
        sample_centroids(data, n_clusters, centroid_sample_options);

    const auto step1_end = Clock::now();
    std::cout << "[build_index] Step 1 done in "
              << format_elapsed(elapsed_ms(step1_start, step1_end))
              << ". Sampled centroids." << std::endl;

    const size_t assign_batch = assignment_batch_size();
    constexpr RotatorType kRotatorType = RotatorType::FhtKacRotator;
    BUILD_CUDA_CHECK(cudaSetDevice(0));
    Rotator<float> *rotator =
        choose_rotator_with_seed<float>(d, kRotatorType, PADDED_DIM, kDefaultRotatorSeed);

    doc_format::Header header;
    header.n = n;
    header.d = d;
    header.n_clusters = n_clusters;
    header.ex_bits = ex_bits;
    header.padded_dim = PADDED_DIM;
    header.rotator_type = kRotatorType;

    const bool use_merged_quantization =
        options.quantization_backend == QuantizationBackend::GpuMerged;
    std::unique_ptr<QuantizedOutputBuffer> quantized_buffer;
    std::unique_ptr<ExSidecarWriter> ex_sidecar_writer;
    std::vector<std::unique_ptr<AssignmentBatchGpuQuantizer>> merged_quantizers;
    AssignmentBatchCallback assignment_batch_callback;
    if (use_merged_quantization)
    {
        const auto device_ids = visible_gpu_ids();
        quantized_buffer = std::make_unique<QuantizedOutputBuffer>(n, ex_bits);
        if (write_ex_sidecar)
        {
            auto ex_sidecar_header = header;
            ex_sidecar_header.ex_bits = sidecar_ex_bits;
            ex_sidecar_writer = std::make_unique<ExSidecarWriter>(
                doc_4bit_ex_path,
                ex_sidecar_header,
                *rotator);
        }
        int max_device_id = -1;
        for (int device_id : device_ids)
        {
            max_device_id = std::max(max_device_id, device_id);
        }
        merged_quantizers.resize(static_cast<size_t>(max_device_id + 1));
        for (int device_id : device_ids)
        {
            merged_quantizers[static_cast<size_t>(device_id)] =
                std::make_unique<AssignmentBatchGpuQuantizer>(
                    device_id,
                    assign_batch,
                    d,
                    ex_bits,
                    sidecar_ex_bits,
                    *rotator,
                    *quantized_buffer,
                    ex_sidecar_writer.get());
        }
        assignment_batch_callback = [&](const AssignmentBatch& batch)
        {
            if (batch.device_id < 0 ||
                static_cast<size_t>(batch.device_id) >= merged_quantizers.size() ||
                !merged_quantizers[static_cast<size_t>(batch.device_id)])
            {
                throw std::runtime_error(
                    "assignment batch references a GPU without a merged quantizer");
            }
            merged_quantizers[static_cast<size_t>(batch.device_id)]->quantize(batch);
        };
        std::cout << "[build_index] Step 2 will quantize assignment batches "
                  << "with quantization_backend=gpu-merged." << std::endl;
    }

    std::cout << "[build_index] Step 2: assignment_batch_size="
              << assign_batch << "." << std::endl;
    const auto step2_start = Clock::now();
    AssignmentTiming assignment_timing;
    auto list_nos = assign_embeddings_multi_gpu(
        data,
        centroids,
        n_clusters,
        assign_batch,
        &assignment_timing,
        assignment_batch_callback);
    const auto step2_end = Clock::now();
    std::cout << "[build_index] Step 2 done in "
              << format_elapsed(elapsed_ms(step2_start, step2_end))
              << ". Embeddings assigned with non-rotated CAGRA." << std::endl;
    std::cout << "[build_index] Step 2a done in "
              << format_elapsed(assignment_timing.cagra_build_ms)
              << ". Temporary per-GPU CAGRA construction max wall time."
              << std::endl;
    std::cout << "[build_index] Step 2b done in "
              << format_elapsed(assignment_timing.search_ms)
              << ". Token-to-centroid CAGRA search max worker wall time."
              << std::endl;
    std::cout << "[build_index] Step 2 throughput: "
              << (elapsed_ms(step2_start, step2_end) > 0.0 ?
                  (static_cast<double>(n) * 1000.0 / elapsed_ms(step2_start, step2_end)) :
                  0.0)
              << " embeddings/s, batches="
              << assignment_timing.launched_batches
              << "." << std::endl;
    if (!merged_quantizers.empty())
    {
        BatchQuantizationTiming merged_timing;
        for (const auto& quantizer : merged_quantizers)
        {
            if (!quantizer)
            {
                continue;
            }
            const auto& timing = quantizer->timing();
            merged_timing.rotate_ms += timing.rotate_ms;
            merged_timing.h2d_ms += timing.h2d_ms;
            merged_timing.encode_ms += timing.encode_ms;
            merged_timing.d2h_ms += timing.d2h_ms;
            merged_timing.copy_ms += timing.copy_ms;
            merged_timing.sidecar_ms += timing.sidecar_ms;
            merged_timing.batches += timing.batches;
            merged_timing.embeddings += timing.embeddings;
        }
        std::cout << "[build_index] Step 2c merged quantization: "
                  << "batches=" << merged_timing.batches
                  << ", embeddings=" << merged_timing.embeddings
                  << ", rotate_ms=" << merged_timing.rotate_ms
                  << ", gpu_h2d_ms=" << merged_timing.h2d_ms
                  << ", gpu_encode_ms=" << merged_timing.encode_ms
                  << ", gpu_d2h_ms=" << merged_timing.d2h_ms
                  << ", main_buffer_copy_ms=" << merged_timing.copy_ms
                  << ", ex_sidecar_write_batch_ms=" << merged_timing.sidecar_ms
                  << "." << std::endl;
    }
    if (exit_after_step2())
    {
        const auto build_end = Clock::now();
        std::cout << "[build_index] Exiting after Step 2 by request "
                  << "(CHIMERA_BUILD_EXIT_AFTER_STEP2=1)." << std::endl;
        std::cout << "[build_index][profile] Total build time: "
                  << format_elapsed(elapsed_ms(build_start, build_end))
                  << std::endl;
        return;
    }
    std::cout << "[build_index] Step 3: Rotating centroids and building persisted CAGRA graph on "
              << n_clusters << " centroids ..." << std::endl;
    const auto step3_start = Clock::now();

    std::vector<float> rotated_centroids(n_clusters * PADDED_DIM);
    for (size_t i = 0; i < n_clusters; ++i)
    {
        rotator->rotate(&centroids[i * d], &rotated_centroids[i * PADDED_DIM]);
    }

    PG_CAGRA *pg_cagra = new PG_CAGRA(n_clusters, PADDED_DIM);
    pg_cagra->build_index(rotated_centroids.data());

    if (output_selection.write_centroids_hnsw)
    {
        const auto step3b_start = Clock::now();
        std::cout << "[build_index] Step 3b: Building hnswlib HNSW on rotated centroids ..." << std::endl;
        build_centroid_hnsw_from_host_data(
            rotated_centroids,
            n_clusters,
            PADDED_DIM,
            centroids_hnsw_path);
        const auto step3b_end = Clock::now();
        std::cout << "[build_index] Step 3b done in "
                  << format_elapsed(elapsed_ms(step3b_start, step3b_end))
                  << ". Saved " << centroids_hnsw_path << "." << std::endl;
    }
    else
    {
        std::cout << "[build_index] Step 3b skipped for build_mode="
                  << build_output_mode_name(options.output_mode) << "." << std::endl;
    }

    const auto step3_end = Clock::now();
    std::cout << "[build_index] Step 3 done in "
              << format_elapsed(elapsed_ms(step3_start, step3_end))
              << ". Persisted rotated CAGRA graph built." << std::endl;

    std::cout << "[build_index] Step 4: Assembling IVF_PG ..." << std::endl;
    const auto step4_start = Clock::now();

    IVF_PG *ivf = new IVF_PG(n_clusters, d, PGType::CAGRA);
    delete ivf->pg_index;
    ivf->pg_index = pg_cagra;
    ivf->build_from_assignments(list_nos.data(), n);
    ivf->save(ivf_path, centroids_path);
    const int max_cluster_size_tokens = ivf->max_cluster_size();
    centroids.clear();
    centroids.shrink_to_fit();
    rotated_centroids.clear();
    rotated_centroids.shrink_to_fit();

    const auto step4_end = Clock::now();
    std::cout << "[build_index] Step 4 done in "
              << format_elapsed(elapsed_ms(step4_start, step4_end))
              << ". IVF_PG assembled and saved. max_cluster_size_tokens="
              << max_cluster_size_tokens << "." << std::endl;

    std::cout << "[build_index] Step 5: Quantizing " << n << " vectors ..." << std::endl;

    std::cout << "[build_index] Step 6: Saving quantized payload to "
              << doc_1bit_path << " and " << doc_4bit_path << " ..." << std::endl;

    const auto step5_start = Clock::now();
    ProgressState step5_progress;
    step5_progress.step_label = "Step 5";
    step5_progress.status_label = "embeddings quantized";
    step5_progress.total_items = n;
    step5_progress.start_time = step5_start;

    if (!quantized_buffer)
    {
        quantized_buffer = std::make_unique<QuantizedOutputBuffer>(n, ex_bits);
    }
    if (write_ex_sidecar && !ex_sidecar_writer)
    {
        auto ex_sidecar_header = header;
        ex_sidecar_header.ex_bits = sidecar_ex_bits;
        ex_sidecar_writer = std::make_unique<ExSidecarWriter>(
            doc_4bit_ex_path,
            ex_sidecar_header,
            *rotator);
    }
    ClusteredStage1Buffer clustered_stage1_buffer(
        ivf->inv_list.size(),
        PADDED_DIM / 8);

    std::vector<int> doc_ptrs(doc_lens.size() + 1, 0);
    for (size_t i = 0; i < doc_lens.size(); ++i)
    {
        doc_ptrs[i + 1] = doc_ptrs[i] + doc_lens[i];
    }
    if (static_cast<size_t>(doc_ptrs.back()) != n)
    {
        throw std::runtime_error(
            "doclens token count does not match embedding count while building clustered sidecar");
    }

    for (size_t pos = 0; pos < ivf->inv_list.size(); ++pos)
    {
        const uint32_t orig_id = ivf->inv_list[pos];
        list_nos[orig_id] = static_cast<uint32_t>(pos);
    }

    constexpr size_t kTargetInputBatchBytes = 64ULL * 1024ULL * 1024ULL;
    const size_t raw_bytes_per_vector = std::max<size_t>(1, d * sizeof(float));
    const size_t batch_size =
        std::max<size_t>(1, kTargetInputBatchBytes / raw_bytes_per_vector);
    std::vector<float> rotated(batch_size * PADDED_DIM);
    std::vector<uint64_t> one_bit_code_batch(batch_size * PADDED_DIM / 64);
    std::vector<uint8_t> full_code_batch(batch_size * PADDED_DIM * (1 + ex_bits) / 8);
    std::vector<uint8_t> ex_code_batch(
        write_ex_sidecar ?
        batch_size * PADDED_DIM * sidecar_ex_bits / 8 :
        0);
    std::vector<float> one_bit_factor_batch(batch_size);
    std::vector<float> ex_factor_batch(batch_size);
    std::vector<float> ex_sidecar_factor_batch(
        write_ex_sidecar ? batch_size : 0);
    const bool use_gpu_quantization =
        options.quantization_backend == QuantizationBackend::Gpu;
    PinnedHostBuffer<uint64_t> pinned_one_bit_code_batch(
        use_gpu_quantization ? one_bit_code_batch.size() : 0);
    PinnedHostBuffer<uint8_t> pinned_full_code_batch(
        use_gpu_quantization ? full_code_batch.size() : 0);
    PinnedHostBuffer<uint8_t> pinned_ex_code_batch(
        use_gpu_quantization ? ex_code_batch.size() : 0);
    PinnedHostBuffer<float> pinned_one_bit_factor_batch(
        use_gpu_quantization ? one_bit_factor_batch.size() : 0);
    PinnedHostBuffer<float> pinned_ex_factor_batch(
        use_gpu_quantization ? ex_factor_batch.size() : 0);
    PinnedHostBuffer<float> pinned_ex_sidecar_factor_batch(
        use_gpu_quantization ? ex_sidecar_factor_batch.size() : 0);
    cudaStream_t quant_stream = nullptr;
    float* d_raw_batch = nullptr;
    uint8_t* d_fht_flip = nullptr;
    size_t fht_flip_bytes = 0;
    uint64_t* d_one_bit_code_batch = nullptr;
    uint8_t* d_full_code_batch = nullptr;
    float* d_one_bit_factor_batch = nullptr;
    float* d_ex_factor_batch = nullptr;
    uint8_t* d_ex_code_batch = nullptr;
    float* d_ex_sidecar_factor_batch = nullptr;
    if (use_gpu_quantization)
    {
        const auto* fht_rotator =
            dynamic_cast<const rotator_impl::FhtKacRotator*>(rotator);
        if (fht_rotator == nullptr || d != PADDED_DIM)
        {
            throw std::runtime_error(
                "quantization_backend=gpu currently requires FhtKacRotator with dim=128");
        }
        BUILD_CUDA_CHECK(cudaSetDevice(0));
        BUILD_CUDA_CHECK(cudaStreamCreate(&quant_stream));
        BUILD_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_raw_batch),
            batch_size * d * sizeof(float)));
        BUILD_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_fht_flip),
            fht_rotator->flip_bytes()));
        BUILD_CUDA_CHECK(cudaMemcpyAsync(
            d_fht_flip,
            fht_rotator->flip_data(),
            fht_rotator->flip_bytes(),
            cudaMemcpyHostToDevice,
            quant_stream));
        BUILD_CUDA_CHECK(cudaStreamSynchronize(quant_stream));
        fht_flip_bytes = fht_rotator->flip_bytes();
        BUILD_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_one_bit_code_batch),
            one_bit_code_batch.size() * sizeof(uint64_t)));
        BUILD_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_full_code_batch),
            full_code_batch.size() * sizeof(uint8_t)));
        BUILD_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_one_bit_factor_batch),
            one_bit_factor_batch.size() * sizeof(float)));
        BUILD_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_ex_factor_batch),
            ex_factor_batch.size() * sizeof(float)));
        if (write_ex_sidecar)
        {
            BUILD_CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_ex_code_batch),
                ex_code_batch.size() * sizeof(uint8_t)));
            BUILD_CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_ex_sidecar_factor_batch),
                ex_sidecar_factor_batch.size() * sizeof(float)));
        }
    }
    double quant_prefetch_ms = 0.0;
    double quant_rotate_ms = 0.0;
    double quant_encode_ms = 0.0;
    double quant_gpu_h2d_ms = 0.0;
    double quant_gpu_d2h_ms = 0.0;
    double quant_copy_ms = 0.0;
    double quant_ex_sidecar_ms = 0.0;
    double quant_cluster_scatter_ms = 0.0;
    if (n > 0 && !use_merged_quantization)
    {
        const auto prefetch_start = Clock::now();
        data.prefetch_embeddings(0, std::min(batch_size, n));
        quant_prefetch_ms += elapsed_ms(prefetch_start, Clock::now());
    }
    for (size_t start = 0; start < n; start += batch_size)
    {
        size_t end = std::min(start + batch_size, n);
        size_t cur_batch = end - start;

        if (use_merged_quantization)
        {
            const auto scatter_start = Clock::now();
            clustered_stage1_buffer.scatter_batch(
                start,
                cur_batch,
                quantized_buffer->one_bit_code_data(start),
                quantized_buffer->one_bit_factor_data(start),
                list_nos.data(),
                doc_ptrs);
            quant_cluster_scatter_ms += elapsed_ms(scatter_start, Clock::now());
            report_progress(step5_progress, cur_batch);
            continue;
        }

        if (end < n)
        {
            const auto prefetch_start = Clock::now();
            data.prefetch_embeddings(end, std::min(batch_size, n - end));
            quant_prefetch_ms += elapsed_ms(prefetch_start, Clock::now());
        }

        const float* raw_batch = data.embedding_ptr(start);
        const uint64_t* one_bit_code_output = one_bit_code_batch.data();
        const uint8_t* full_code_output = full_code_batch.data();
        const float* one_bit_factor_output = one_bit_factor_batch.data();
        const float* ex_factor_output = ex_factor_batch.data();
        const uint8_t* ex_code_output = ex_code_batch.data();
        const float* ex_sidecar_factor_output = ex_sidecar_factor_batch.data();
        if (use_gpu_quantization)
        {
            const size_t raw_bytes = cur_batch * d * sizeof(float);
            const size_t one_bit_code_bytes =
                cur_batch * (PADDED_DIM / 64) * sizeof(uint64_t);
            const size_t full_code_bytes =
                cur_batch * PADDED_DIM * (1 + ex_bits) / 8;
            const size_t sidecar_code_bytes =
                write_ex_sidecar ? cur_batch * PADDED_DIM * sidecar_ex_bits / 8 : 0;

            const auto h2d_start = Clock::now();
            BUILD_CUDA_CHECK(cudaMemcpyAsync(
                d_raw_batch,
                raw_batch,
                raw_bytes,
                cudaMemcpyHostToDevice,
                quant_stream));
            BUILD_CUDA_CHECK(cudaStreamSynchronize(quant_stream));
            quant_gpu_h2d_ms += elapsed_ms(h2d_start, Clock::now());

            GpuRotateQuantizeBatchParams params;
            params.d_raw = d_raw_batch;
            params.count = cur_batch;
            params.dim = d;
            params.d_fht_flip = d_fht_flip;
            params.fht_flip_bytes = fht_flip_bytes;
            params.d_one_bit_code = d_one_bit_code_batch;
            params.d_one_bit_factor = d_one_bit_factor_batch;
            params.full_ex_bits = ex_bits;
            params.d_full_code = d_full_code_batch;
            params.d_full_factor = d_ex_factor_batch;
            params.sidecar_ex_bits = sidecar_ex_bits;
            params.d_sidecar_code = d_ex_code_batch;
            params.d_sidecar_factor = d_ex_sidecar_factor_batch;
            params.stream = quant_stream;

            const auto encode_start = Clock::now();
            gpu_rotate_quantize_batch(params);
            BUILD_CUDA_CHECK(cudaStreamSynchronize(quant_stream));
            quant_encode_ms += elapsed_ms(encode_start, Clock::now());

            const auto d2h_start = Clock::now();
            BUILD_CUDA_CHECK(cudaMemcpyAsync(
                pinned_one_bit_code_batch.data(),
                d_one_bit_code_batch,
                one_bit_code_bytes,
                cudaMemcpyDeviceToHost,
                quant_stream));
            BUILD_CUDA_CHECK(cudaMemcpyAsync(
                pinned_full_code_batch.data(),
                d_full_code_batch,
                full_code_bytes,
                cudaMemcpyDeviceToHost,
                quant_stream));
            BUILD_CUDA_CHECK(cudaMemcpyAsync(
                pinned_one_bit_factor_batch.data(),
                d_one_bit_factor_batch,
                cur_batch * sizeof(float),
                cudaMemcpyDeviceToHost,
                quant_stream));
            BUILD_CUDA_CHECK(cudaMemcpyAsync(
                pinned_ex_factor_batch.data(),
                d_ex_factor_batch,
                cur_batch * sizeof(float),
                cudaMemcpyDeviceToHost,
                quant_stream));
            if (write_ex_sidecar)
            {
                BUILD_CUDA_CHECK(cudaMemcpyAsync(
                    pinned_ex_code_batch.data(),
                    d_ex_code_batch,
                    sidecar_code_bytes,
                    cudaMemcpyDeviceToHost,
                    quant_stream));
                BUILD_CUDA_CHECK(cudaMemcpyAsync(
                    pinned_ex_sidecar_factor_batch.data(),
                    d_ex_sidecar_factor_batch,
                    cur_batch * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    quant_stream));
            }
            BUILD_CUDA_CHECK(cudaStreamSynchronize(quant_stream));
            quant_gpu_d2h_ms += elapsed_ms(d2h_start, Clock::now());
            one_bit_code_output = pinned_one_bit_code_batch.data();
            full_code_output = pinned_full_code_batch.data();
            one_bit_factor_output = pinned_one_bit_factor_batch.data();
            ex_factor_output = pinned_ex_factor_batch.data();
            ex_code_output = pinned_ex_code_batch.data();
            ex_sidecar_factor_output = pinned_ex_sidecar_factor_batch.data();
        }
        else
        {
            const auto encode_start = Clock::now();
#pragma omp parallel for
            for (size_t i = 0; i < cur_batch; ++i)
            {
                const float* raw_vec = raw_batch + i * d;
                rotator->rotate(raw_vec, &rotated[i * PADDED_DIM]);

                encode_one_bit(
                    &rotated[i * PADDED_DIM],
                    PADDED_DIM,
                    one_bit_code_batch.data() + i * (PADDED_DIM / 64),
                    &one_bit_factor_batch[i]);

                encode_full_code(
                    &rotated[i * PADDED_DIM],
                    PADDED_DIM,
                    ex_bits,
                    full_code_batch.data() + i * (PADDED_DIM * (1 + ex_bits) / 8),
                    &ex_factor_batch[i]);

                if (write_ex_sidecar)
                {
                    encode_ex_bits(
                        &rotated[i * PADDED_DIM],
                        PADDED_DIM,
                        sidecar_ex_bits,
                        ex_code_batch.data() + i * (PADDED_DIM * sidecar_ex_bits / 8),
                        &ex_sidecar_factor_batch[i]);
                }
            }
            quant_encode_ms += elapsed_ms(encode_start, Clock::now());
        }
        const auto copy_start = Clock::now();
        quantized_buffer->copy_batch(
            start,
            cur_batch,
            one_bit_code_output,
            full_code_output,
            one_bit_factor_output,
            ex_factor_output);
        quant_copy_ms += elapsed_ms(copy_start, Clock::now());
        if (ex_sidecar_writer)
        {
            const auto sidecar_start = Clock::now();
            ex_sidecar_writer->write_batch(
                start,
                cur_batch,
                ex_code_output,
                ex_sidecar_factor_output);
            quant_ex_sidecar_ms += elapsed_ms(sidecar_start, Clock::now());
        }
        const auto scatter_start = Clock::now();
        clustered_stage1_buffer.scatter_batch(
            start,
            cur_batch,
            one_bit_code_output,
            one_bit_factor_output,
            list_nos.data(),
            doc_ptrs);
        quant_cluster_scatter_ms += elapsed_ms(scatter_start, Clock::now());
        report_progress(step5_progress, cur_batch);
    }
    if (use_gpu_quantization)
    {
        BUILD_CUDA_CHECK(cudaFree(d_raw_batch));
        BUILD_CUDA_CHECK(cudaFree(d_fht_flip));
        BUILD_CUDA_CHECK(cudaFree(d_one_bit_code_batch));
        BUILD_CUDA_CHECK(cudaFree(d_full_code_batch));
        BUILD_CUDA_CHECK(cudaFree(d_one_bit_factor_batch));
        BUILD_CUDA_CHECK(cudaFree(d_ex_factor_batch));
        if (d_ex_code_batch != nullptr)
        {
            BUILD_CUDA_CHECK(cudaFree(d_ex_code_batch));
        }
        if (d_ex_sidecar_factor_batch != nullptr)
        {
            BUILD_CUDA_CHECK(cudaFree(d_ex_sidecar_factor_batch));
        }
        BUILD_CUDA_CHECK(cudaStreamDestroy(quant_stream));
    }

    const auto step5_end = Clock::now();
    const auto write_quantized_start = Clock::now();
    quantized_buffer->write_to_files(
        doc_1bit_path,
        doc_4bit_path,
        header,
        *rotator);
    const auto write_quantized_end = Clock::now();
    const auto write_clustered_start = Clock::now();
    clustered_stage1_buffer.write_to_file(clustered_stage1_path);
    const auto write_clustered_end = Clock::now();
    auto write_metadata_start = write_clustered_end;
    auto write_metadata_end = write_clustered_end;
    if (output_selection.write_metadata)
    {
        write_metadata_start = Clock::now();
        write_index_metadata_json(
            index_dir,
            header,
            max_cluster_size_tokens,
            centroid_sample_options,
            options.quantization_mode,
            sidecar_ex_bits,
            options.quantization_backend);
        write_metadata_end = Clock::now();
    }
    const auto flush_end = Clock::now();

    std::cout << "[build_index] Step 5 done in "
              << format_elapsed(elapsed_ms(step5_start, step5_end))
              << ". Quantization complete." << std::endl;
    std::cout << "[build_index] Step 5 profile: prefetch_ms="
              << quant_prefetch_ms
              << ", rotate_ms=" << quant_rotate_ms
              << ", encode_ms=" << quant_encode_ms
              << ", gpu_h2d_ms=" << quant_gpu_h2d_ms
              << ", gpu_d2h_ms=" << quant_gpu_d2h_ms
              << ", main_buffer_copy_ms=" << quant_copy_ms
              << ", ex_sidecar_write_batch_ms=" << quant_ex_sidecar_ms
              << ", cluster_scatter_ms=" << quant_cluster_scatter_ms
              << "." << std::endl;
    std::cout << "[build_index] Step 6 done in "
              << format_elapsed(elapsed_ms(write_quantized_start, write_quantized_end))
              << ". Quantized payload saved to " << doc_1bit_path
              << " and " << doc_4bit_path
              << "." << std::endl;
    if (write_ex_sidecar)
    {
        std::cout << "[build_index] Step 6b done during Step 5 streaming batches."
                  << " Residual sidecar saved to " << doc_4bit_ex_path
                  << "." << std::endl;
    }
    else
    {
        std::cout << "[build_index] Step 6b skipped for quantization="
                  << quantization_mode_name(options.quantization_mode)
                  << "." << std::endl;
    }
    std::cout << "[build_index] Step 6c done in "
              << format_elapsed(elapsed_ms(write_clustered_start, write_clustered_end))
              << ". Clustered stage-1 payload saved to " << clustered_stage1_path
              << "." << std::endl;
    if (output_selection.write_metadata)
    {
        std::cout << "[build_index] Step 6d done in "
                  << format_elapsed(elapsed_ms(write_metadata_start, write_metadata_end))
                  << ". Metadata saved to " << metadata_path
                  << "." << std::endl;
    }
    else
    {
        std::cout << "[build_index] Step 6d skipped for build_mode="
                  << build_output_mode_name(options.output_mode) << "." << std::endl;
    }
    std::cout << "[build_index] Quantize pipeline total: "
              << format_elapsed(elapsed_ms(step5_start, step5_end)) << "." << std::endl;
    delete ivf;
    ivf = nullptr;
    list_nos.clear();
    list_nos.shrink_to_fit();
    delete rotator;

    const auto build_end = Clock::now();
    std::cout << "[build_index][profile] Total build time: "
              << format_elapsed(elapsed_ms(build_start, build_end)) << std::endl;

    std::cout << "[build_index] Done. Index saved under: " << index_dir << std::endl;
}


}  // namespace Chimera
