#include "build_gpu_index.hpp"

#define EIGEN_NO_CUDA

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <sys/mman.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include "gpu_index_layout.hpp"
#include "clustered_stage1_file_format.hpp"
#include "mvr_index_file_format.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "gpu_config.cuh"
#include "ivf_pg.hpp"
#include "quantization.hpp"

using namespace rabitqlib;

namespace {

void build_cuda_check(cudaError_t status, const char* expr, const char* file, int line)
{
    if (status == cudaSuccess)
    {
        return;
    }

    throw std::runtime_error(
        std::string("CUDA error at ") + file + ":" + std::to_string(line) +
        " for " + expr + ": " + cudaGetErrorString(status));
}

#define BUILD_CUDA_CHECK(call) \
    build_cuda_check((call), #call, __FILE__, __LINE__)

double elapsed_ms(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string format_elapsed(double milliseconds)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);

    constexpr double kMsPerSecond = 1000.0;
    constexpr double kMsPerMinute = 60.0 * kMsPerSecond;
    constexpr double kMsPerHour = 60.0 * kMsPerMinute;

    if (milliseconds >= kMsPerHour)
    {
        out << (milliseconds / kMsPerHour) << " hours";
    }
    else if (milliseconds >= kMsPerMinute)
    {
        out << (milliseconds / kMsPerMinute) << " minutes";
    }
    else
    {
        out << (milliseconds / kMsPerSecond) << " seconds";
    }

    return out.str();
}

std::vector<int> visible_gpu_ids()
{
    int device_count = 0;
    BUILD_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0)
    {
        throw std::runtime_error("No CUDA devices are visible for gpu_build");
    }

    std::vector<int> device_ids(static_cast<size_t>(device_count));
    std::iota(device_ids.begin(), device_ids.end(), 0);
    return device_ids;
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
        const mvr_index_file_format::Header& header,
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

        mvr_index_file_format::write_header(output, header, path_);
        mvr_index_file_format::save_rotator(output, rotator, path_);

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

    void write_to_file(
        const std::string& path,
        const mvr_index_file_format::Header& header,
        const Rotator<float>& rotator) const
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output.is_open())
        {
            throw std::runtime_error("Failed to open quantized output file: " + path);
        }

        mvr_index_file_format::write_header(output, header, path);
        mvr_index_file_format::save_rotator(output, rotator, path);
        output.write(
            reinterpret_cast<const char*>(one_bit_code_.data()),
            static_cast<std::streamsize>(one_bit_code_.size() * sizeof(uint64_t)));
        output.write(
            reinterpret_cast<const char*>(full_code_.data()),
            static_cast<std::streamsize>(full_code_.size()));
        output.write(
            reinterpret_cast<const char*>(one_bit_factor_.data()),
            static_cast<std::streamsize>(one_bit_factor_.size() * sizeof(float)));
        output.write(
            reinterpret_cast<const char*>(ex_factor_.data()),
            static_cast<std::streamsize>(ex_factor_.size() * sizeof(float)));
        if (!output)
        {
            throw std::runtime_error("Failed to write quantized output file: " + path);
        }
    }

   private:
    std::vector<uint64_t> one_bit_code_;
    std::vector<uint8_t> full_code_;
    std::vector<float> one_bit_factor_;
    std::vector<float> ex_factor_;
    size_t one_bit_bytes_per_vector_ = 0;
    size_t full_code_bytes_per_vector_ = 0;
};

class ClusteredStage1OutputWriter
{
   public:
    ClusteredStage1OutputWriter(
        const std::string& path,
        size_t n_entries,
        size_t code_bytes_per_vector)
        : path_(path)
    {
        clustered_stage1_file_format::Header header;
        header.n_entries = n_entries;
        header.code_bytes_per_vector = code_bytes_per_vector;

        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        if (!output.is_open())
        {
            throw std::runtime_error("Failed to open clustered sidecar output file: " + path_);
        }

        clustered_stage1_file_format::write_header(output, header, path_);

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
        clustered_stage1_file_format::Header header;
        header.n_entries = factor_.size();
        header.code_bytes_per_vector = code_bytes_per_vector_;

        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output.is_open())
        {
            throw std::runtime_error("Failed to open clustered stage-1 output file: " + path);
        }

        clustered_stage1_file_format::write_header(output, header, path);
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

struct Step2ProgressState
{
    size_t total_documents = 0;
    std::atomic<size_t> handled_documents {0};
    std::atomic<size_t> next_percent_to_report {1};
    std::mutex output_mutex;
};

void report_step2_progress(Step2ProgressState& progress, size_t handled_now)
{
    if (progress.total_documents == 0 || handled_now == 0)
    {
        return;
    }

    const size_t handled =
        progress.handled_documents.fetch_add(handled_now, std::memory_order_relaxed) +
        handled_now;
    const size_t percent =
        std::min<size_t>(100, handled * 100 / progress.total_documents);

    size_t next_percent =
        progress.next_percent_to_report.load(std::memory_order_relaxed);
    while (percent >= next_percent)
    {
        if (progress.next_percent_to_report.compare_exchange_weak(
                next_percent,
                percent + 1,
                std::memory_order_relaxed,
                std::memory_order_relaxed))
        {
            std::lock_guard<std::mutex> lock(progress.output_mutex);
            std::cout << "[build_index] Step 2 progress: " << percent
                      << "% documents handled." << std::endl;
            return;
        }
    }
}

void assign_shard_batches_on_device(
    const MmapedEmbeddings& data,
    const std::vector<float>& centroids,
    size_t n_clusters,
    size_t assign_batch,
    int device_id,
    std::atomic<size_t>& next_batch_start,
    std::atomic<bool>& should_stop,
    std::vector<uint32_t>& list_nos,
    Step2ProgressState& progress)
{
    BUILD_CUDA_CHECK(cudaSetDevice(device_id));

    struct BufferSlot
    {
        cudaStream_t stream = nullptr;
        float* d_q_ptr = nullptr;
        uint32_t* d_lab_ptr = nullptr;
        float* d_dist_ptr = nullptr;
        float* h_q_ptr = nullptr;
        uint32_t* h_labels = nullptr;
        size_t start = 0;
        size_t cur = 0;
        bool in_flight = false;
    };

    constexpr size_t kBufferCount = 2;
    std::vector<BufferSlot> buffers(kBufferCount);

    auto cleanup = [&]()
    {
        for (auto& buffer : buffers)
        {
            if (buffer.h_labels != nullptr)
            {
                cudaFreeHost(buffer.h_labels);
            }
            if (buffer.h_q_ptr != nullptr)
            {
                cudaFreeHost(buffer.h_q_ptr);
            }
            if (buffer.d_dist_ptr != nullptr)
            {
                cudaFree(buffer.d_dist_ptr);
            }
            if (buffer.d_lab_ptr != nullptr)
            {
                cudaFree(buffer.d_lab_ptr);
            }
            if (buffer.d_q_ptr != nullptr)
            {
                cudaFree(buffer.d_q_ptr);
            }
            if (buffer.stream != nullptr)
            {
                cudaStreamDestroy(buffer.stream);
            }
        }
    };

    try
    {
        PG_CAGRA assignment_cagra(n_clusters, data.d);
        assignment_cagra.build_index(centroids.data());

        for (auto& buffer : buffers)
        {
            BUILD_CUDA_CHECK(cudaStreamCreate(&buffer.stream));
            BUILD_CUDA_CHECK(cudaMalloc(&buffer.d_q_ptr, assign_batch * data.d * sizeof(float)));
            BUILD_CUDA_CHECK(cudaMalloc(&buffer.d_lab_ptr, assign_batch * sizeof(uint32_t)));
            BUILD_CUDA_CHECK(cudaMalloc(&buffer.d_dist_ptr, assign_batch * sizeof(float)));
            BUILD_CUDA_CHECK(
                cudaMallocHost(&buffer.h_q_ptr, assign_batch * data.d * sizeof(float)));
            BUILD_CUDA_CHECK(cudaMallocHost(&buffer.h_labels, assign_batch * sizeof(uint32_t)));
        }

        auto finalize_buffer = [&](BufferSlot& buffer)
        {
            if (!buffer.in_flight)
            {
                return;
            }

            BUILD_CUDA_CHECK(cudaStreamSynchronize(buffer.stream));
            std::memcpy(
                list_nos.data() + buffer.start,
                buffer.h_labels,
                buffer.cur * sizeof(uint32_t));
            report_step2_progress(progress, buffer.cur);
            buffer.in_flight = false;
        };

        auto launch_buffer = [&](BufferSlot& buffer, size_t start, size_t cur)
        {
            data.copy_embeddings(start, cur, buffer.h_q_ptr);

            BUILD_CUDA_CHECK(cudaMemcpyAsync(
                buffer.d_q_ptr,
                buffer.h_q_ptr,
                cur * data.d * sizeof(float),
                cudaMemcpyHostToDevice,
                buffer.stream));

            assignment_cagra.search_batch_gpu(
                buffer.d_q_ptr,
                cur,
                1,
                buffer.d_dist_ptr,
                buffer.d_lab_ptr,
                buffer.stream);

            BUILD_CUDA_CHECK(cudaMemcpyAsync(
                buffer.h_labels,
                buffer.d_lab_ptr,
                cur * sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                buffer.stream));

            buffer.start = start;
            buffer.cur = cur;
            buffer.in_flight = true;
        };

        size_t next_buffer = 0;
        while (!should_stop.load(std::memory_order_relaxed))
        {
            BufferSlot& buffer = buffers[next_buffer];
            finalize_buffer(buffer);

            const size_t start =
                next_batch_start.fetch_add(assign_batch, std::memory_order_relaxed);
            if (start >= data.num_embeddings)
            {
                break;
            }

            const size_t cur = std::min(assign_batch, data.num_embeddings - start);
            launch_buffer(buffer, start, cur);
            next_buffer = (next_buffer + 1) % kBufferCount;
        }

        for (auto& buffer : buffers)
        {
            finalize_buffer(buffer);
        }
    }
    catch (...)
    {
        cleanup();
        throw;
    }

    cleanup();
}

std::vector<uint32_t> assign_embeddings_multi_gpu(
    const MmapedEmbeddings& data,
    const std::vector<float>& centroids,
    size_t n_clusters,
    size_t assign_batch)
{
    auto device_ids = visible_gpu_ids();
    const size_t total_batches = (data.num_embeddings + assign_batch - 1) / assign_batch;
    const size_t worker_count = std::min(device_ids.size(), total_batches);
    device_ids.resize(worker_count);

    std::cout << "[build_index] Step 2: Assigning embeddings with " << worker_count
              << " visible GPU(s)." << std::endl;

    std::vector<uint32_t> list_nos(data.num_embeddings);
    if (worker_count == 0)
    {
        return list_nos;
    }

    std::atomic<size_t> next_batch_start {0};
    std::atomic<bool> should_stop {false};
    Step2ProgressState progress;
    progress.total_documents = data.num_embeddings;
    std::mutex error_mutex;
    std::exception_ptr first_error;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (const int device_id : device_ids)
    {
        workers.emplace_back([&, device_id]()
        {
            try
            {
                assign_shard_batches_on_device(
                    data,
                    centroids,
                    n_clusters,
                    assign_batch,
                    device_id,
                    next_batch_start,
                    should_stop,
                    list_nos,
                    progress);
            }
            catch (...)
            {
                should_stop.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error)
                {
                    first_error = std::current_exception();
                }
            }
        });
    }

    for (auto& worker : workers)
    {
        worker.join();
    }

    if (first_error)
    {
        std::rethrow_exception(first_error);
    }

    BUILD_CUDA_CHECK(cudaSetDevice(device_ids.front()));
    return list_nos;
}

}  // namespace

void build_index(
    const MmapedEmbeddings &data,
    size_t n_clusters,
    size_t ex_bits,
    const std::vector<int> &doc_lens,
    const std::string &index_dir)
{
    using Clock = std::chrono::steady_clock;

    const size_t n = data.num_embeddings;
    const size_t d = data.d;
    const auto build_start = Clock::now();

    std::filesystem::create_directories(index_dir);
    std::error_code cleanup_ec;
    std::filesystem::remove(
        gpu_index_layout::legacy_quantized_data_path(index_dir), cleanup_ec);
    cleanup_ec.clear();
    std::filesystem::remove(
        gpu_index_layout::legacy_clustered_stage1_path(index_dir), cleanup_ec);

    const std::string ivf_path = gpu_index_layout::ivf_path(index_dir);
    const std::string quantized_data_path = gpu_index_layout::cpu_index_path(index_dir);
    const std::string centroids_path = gpu_index_layout::centroids_path(index_dir);
    const std::string clustered_stage1_path = gpu_index_layout::gpu_index_path(index_dir);

    std::vector<float> centroids;

    std::cout << "[build_index] Step 1: Randomly sampling " << n_clusters
              << " centroids from " << n << " vectors (d=" << d << ") ..." << std::endl;
    const auto step1_start = Clock::now();

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(n_clusters);

    centroids.resize(n_clusters * d);
    for (size_t i = 0; i < n_clusters; ++i)
    {
        std::memcpy(&centroids[i * d], data.embedding_ptr(indices[i]), d * sizeof(float));
    }

    const auto step1_end = Clock::now();
    std::cout << "[build_index] Step 1 done in "
              << format_elapsed(elapsed_ms(step1_start, step1_end))
              << ". Sampled centroids." << std::endl;

    constexpr size_t assign_batch = 65536;
    const auto step2_start = Clock::now();
    auto list_nos = assign_embeddings_multi_gpu(data, centroids, n_clusters, assign_batch);
    const auto step2_end = Clock::now();
    std::cout << "[build_index] Step 2 done in "
              << format_elapsed(elapsed_ms(step2_start, step2_end))
              << ". Embeddings assigned with non-rotated CAGRA." << std::endl;

    constexpr RotatorType kRotatorType = RotatorType::FhtKacRotator;
    BUILD_CUDA_CHECK(cudaSetDevice(0));
    Rotator<float> *rotator = choose_rotator<float>(d, kRotatorType, PADDED_DIM);

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
    centroids.clear();
    centroids.shrink_to_fit();
    rotated_centroids.clear();
    rotated_centroids.shrink_to_fit();

    const auto step4_end = Clock::now();
    std::cout << "[build_index] Step 4 done in "
              << format_elapsed(elapsed_ms(step4_start, step4_end))
              << ". IVF_PG assembled and saved." << std::endl;

    std::cout << "[build_index] Step 5: Quantizing " << n << " vectors ..." << std::endl;

    mvr_index_file_format::Header header;
    header.n = n;
    header.d = d;
    header.n_clusters = n_clusters;
    header.ex_bits = ex_bits;
    header.padded_dim = PADDED_DIM;
    header.rotator_type = kRotatorType;

    std::cout << "[build_index] Step 6: Saving quantized payload to "
              << quantized_data_path << " ..." << std::endl;

    const auto step5_start = Clock::now();

    QuantizedOutputBuffer quantized_buffer(n, ex_bits);
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

    size_t batch_size = 10240;
    std::vector<float> raw_batch(batch_size * d);
    for (size_t start = 0; start < n; start += batch_size)
    {
        size_t end = std::min(start + batch_size, n);
        size_t cur_batch = end - start;

        data.copy_embeddings(start, cur_batch, raw_batch.data());
        std::vector<float> rotated(cur_batch * PADDED_DIM);
        std::vector<uint64_t> one_bit_code_batch(cur_batch * PADDED_DIM / 64);
        std::vector<uint8_t> full_code_batch(cur_batch * PADDED_DIM * (1 + ex_bits) / 8);
        std::vector<float> one_bit_factor_batch(cur_batch);
        std::vector<float> ex_factor_batch(cur_batch);
#pragma omp parallel for
        for (size_t i = 0; i < cur_batch; ++i)
        {
            rotator->rotate(&raw_batch[i * d], &rotated[i * PADDED_DIM]);

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
        }
        quantized_buffer.copy_batch(
            start,
            cur_batch,
            one_bit_code_batch.data(),
            full_code_batch.data(),
            one_bit_factor_batch.data(),
            ex_factor_batch.data());
        clustered_stage1_buffer.scatter_batch(
            start,
            cur_batch,
            one_bit_code_batch.data(),
            one_bit_factor_batch.data(),
            list_nos.data(),
            doc_ptrs);
    }

    const auto step5_end = Clock::now();
    const auto flush_start = Clock::now();
    quantized_buffer.write_to_file(
        quantized_data_path,
        header,
        *rotator);
    clustered_stage1_buffer.write_to_file(clustered_stage1_path);
    const auto flush_end = Clock::now();

    std::cout << "[build_index] Step 5 done in "
              << format_elapsed(elapsed_ms(step5_start, step5_end))
              << ". Quantization complete." << std::endl;
    std::cout << "[build_index] Step 6 done in "
              << format_elapsed(elapsed_ms(flush_start, flush_end))
              << ". Quantized payload saved to " << quantized_data_path
              << "." << std::endl;
    std::cout << "[build_index] Step 6b done in "
              << format_elapsed(elapsed_ms(flush_start, flush_end))
              << ". Clustered stage-1 payload saved to " << clustered_stage1_path
              << "." << std::endl;
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

void build_clustered_stage1_sidecar(
    const std::vector<int>& doc_lens,
    const std::string& source_index_dir,
    const std::string& index_dir)
{
    using Clock = std::chrono::steady_clock;

    const auto build_start = Clock::now();
    std::filesystem::create_directories(index_dir);
    std::error_code cleanup_ec;
    std::filesystem::remove(
        gpu_index_layout::legacy_quantized_data_path(index_dir), cleanup_ec);
    cleanup_ec.clear();
    std::filesystem::remove(
        gpu_index_layout::legacy_clustered_stage1_path(index_dir), cleanup_ec);

    const auto source_quantized_path =
        gpu_index_layout::resolve_existing_cpu_index_path(source_index_dir);
    const auto source_ivf_path = gpu_index_layout::ivf_path(source_index_dir);
    const auto source_centroids_path = gpu_index_layout::centroids_path(source_index_dir);

    const auto target_quantized_path = gpu_index_layout::cpu_index_path(index_dir);
    const auto target_ivf_path = gpu_index_layout::ivf_path(index_dir);
    const auto target_centroids_path = gpu_index_layout::centroids_path(index_dir);
    const auto target_clustered_stage1_path = gpu_index_layout::gpu_index_path(index_dir);

    std::cout << "[build_index] Fast sidecar mode: cloning base index from "
              << source_index_dir << " to " << index_dir << std::endl;

    const auto copy_start = Clock::now();
    std::filesystem::copy_file(
        source_quantized_path,
        target_quantized_path,
        std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(
        source_ivf_path,
        target_ivf_path,
        std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(
        source_centroids_path,
        target_centroids_path,
        std::filesystem::copy_options::overwrite_existing);
    std::cout << "[build_index] Base index cloned in "
              << format_elapsed(elapsed_ms(copy_start, Clock::now())) << std::endl;

    std::ifstream inf(target_quantized_path, std::ios::binary);
    const auto header = mvr_index_file_format::read_header(inf, target_quantized_path);
    const size_t n = header.n;
    const size_t code_bytes_per_vector = PADDED_DIM / 8;
    if (header.padded_dim != PADDED_DIM) {
        throw std::runtime_error(
            "Index file padded_dim=" + std::to_string(header.padded_dim) +
            " does not match compiled PADDED_DIM=" + std::to_string(PADDED_DIM));
    }

    auto* rotator = mvr_index_file_format::load_rotator(inf, header, target_quantized_path);
    delete rotator;
    rotator = nullptr;
    const auto prefix_bytes = static_cast<size_t>(inf.tellg());

    const size_t one_bit_bytes = n * code_bytes_per_vector;
    const size_t full_code_bytes = n * PADDED_DIM * (1 + header.ex_bits) / 8;
    const size_t factor_offset = prefix_bytes + one_bit_bytes + full_code_bytes;
    inf.close();

    if (doc_lens.empty()) {
        throw std::runtime_error("doclens is empty while generating clustered sidecar");
    }

    std::vector<int> doc_ptrs(doc_lens.size() + 1, 0);
    for (size_t i = 0; i < doc_lens.size(); ++i) {
        doc_ptrs[i + 1] = doc_ptrs[i] + doc_lens[i];
    }
    if (static_cast<size_t>(doc_ptrs.back()) != n) {
        throw std::runtime_error(
            "doclens token count does not match embedding count while generating clustered sidecar");
    }

    IVF_PG ivf(header.n_clusters, header.d, PGType::CAGRA);
    ivf.load(target_ivf_path, target_centroids_path);

    std::vector<uint32_t> cluster_rank(n);
    for (size_t pos = 0; pos < ivf.inv_list.size(); ++pos) {
        cluster_rank[ivf.inv_list[pos]] = static_cast<uint32_t>(pos);
    }

    ClusteredStage1OutputWriter clustered_stage1_writer(
        target_clustered_stage1_path,
        ivf.inv_list.size(),
        code_bytes_per_vector);

    std::ifstream code_file(target_quantized_path, std::ios::binary);
    std::ifstream factor_file(target_quantized_path, std::ios::binary);
    code_file.seekg(static_cast<std::streamoff>(prefix_bytes));
    factor_file.seekg(static_cast<std::streamoff>(factor_offset));
    if (!code_file || !factor_file) {
        throw std::runtime_error(
            "Failed to position cpu index while generating gpu_index.bin");
    }

    const auto sidecar_start = Clock::now();
    const size_t batch_vectors = 1 << 20;
    std::vector<uint64_t> one_bit_code_batch(batch_vectors * PADDED_DIM / 64);
    std::vector<float> one_bit_factor_batch(batch_vectors);

    for (size_t start = 0; start < n; start += batch_vectors) {
        const size_t cur_batch = std::min(batch_vectors, n - start);
        const size_t cur_code_bytes = cur_batch * code_bytes_per_vector;
        const size_t cur_factor_bytes = cur_batch * sizeof(float);

        code_file.read(
            reinterpret_cast<char*>(one_bit_code_batch.data()),
            static_cast<std::streamsize>(cur_code_bytes));
        factor_file.read(
            reinterpret_cast<char*>(one_bit_factor_batch.data()),
            static_cast<std::streamsize>(cur_factor_bytes));
        if (!code_file || !factor_file) {
            throw std::runtime_error(
                "Failed to read one-bit sections while generating clustered sidecar");
        }

        clustered_stage1_writer.scatter_batch(
            start,
            cur_batch,
            one_bit_code_batch.data(),
            one_bit_factor_batch.data(),
            cluster_rank.data(),
            doc_ptrs);
    }

    code_file.close();
    factor_file.close();

    std::cout << "[build_index] gpu_index.bin generated in "
              << format_elapsed(elapsed_ms(sidecar_start, Clock::now())) << std::endl;
    std::cout << "[build_index][profile] Fast sidecar build total: "
              << format_elapsed(elapsed_ms(build_start, Clock::now())) << std::endl;
}
