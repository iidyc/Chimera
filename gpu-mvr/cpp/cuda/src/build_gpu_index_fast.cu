#include "build_gpu_index_fast.hpp"

#define EIGEN_NO_CUDA

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "clustered_stage1_file_format.hpp"
#include "gpu_config.cuh"
#include "gpu_index_layout.hpp"
#include "ivf_pg.hpp"
#include "mvr_index_file_format.hpp"
#include "quantization.hpp"
#include "rabitqlib/utils/rotator.hpp"

using namespace rabitqlib;

namespace {

constexpr std::uint64_t kDefaultCentroidSampleSeed = 42;
constexpr std::uint64_t kDefaultRotatorSeed = 42;

struct SplitQuantizedSectionLayout
{
    size_t prefix_bytes = 0;
    size_t one_bit_offset = 0;
    size_t one_bit_factor_offset = 0;
    size_t ex_factor_offset = 0;
    size_t full_code_offset = 0;
    size_t one_bit_bytes_per_vector = 0;
    size_t full_code_bytes_per_vector = 0;
    size_t doc_1bit_total_bytes = 0;
    size_t doc_4bit_total_bytes = 0;
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

struct BuildMetadataOptions
{
    std::string builder_name = "gpu_build_fast";
    std::optional<std::uint64_t> centroid_sample_seed;
    std::optional<std::uint64_t> rotator_seed;
    bool clustered_only_rebuild = false;
};

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

#define BUILD_FAST_CUDA_CHECK(call) \
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

std::string current_utc_timestamp()
{
    std::time_t now = std::time(nullptr);
    std::tm utc_tm {};
#if defined(_WIN32)
    gmtime_s(&utc_tm, &now);
#else
    gmtime_r(&now, &utc_tm);
#endif
    char buffer[32];
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc_tm) == 0)
    {
        return "unknown";
    }
    return buffer;
}

std::vector<int> visible_gpu_ids()
{
    int device_count = 0;
    BUILD_FAST_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0)
    {
        throw std::runtime_error("No CUDA devices are visible for gpu_build_fast");
    }

    std::vector<int> device_ids(static_cast<size_t>(device_count));
    std::iota(device_ids.begin(), device_ids.end(), 0);
    return device_ids;
}

void validate_doc_lens(const std::vector<int>& doc_lens, size_t expected_tokens)
{
    if (doc_lens.empty())
    {
        throw std::runtime_error("doclens is empty");
    }

    size_t total_tokens = 0;
    for (const int doc_len : doc_lens)
    {
        if (doc_len < 0)
        {
            throw std::runtime_error("doclens contains a negative document length");
        }
        total_tokens += static_cast<size_t>(doc_len);
    }

    if (total_tokens != expected_tokens)
    {
        throw std::runtime_error(
            "doclens token count " + std::to_string(total_tokens) +
            " does not match embedding count " + std::to_string(expected_tokens));
    }
}

std::vector<int> build_doc_ptrs(const std::vector<int>& doc_lens)
{
    std::vector<int> doc_ptrs(doc_lens.size() + 1, 0);
    for (size_t i = 0; i < doc_lens.size(); ++i)
    {
        doc_ptrs[i + 1] = doc_ptrs[i] + doc_lens[i];
    }
    return doc_ptrs;
}

class SplitQuantizedOutputBuffer
{
   public:
    explicit SplitQuantizedOutputBuffer(
        const mvr_index_file_format::Header& header)
        : one_bit_code_(mvr_index_file_format::one_bit_code_bytes(header) / sizeof(uint64_t)),
          full_code_(mvr_index_file_format::full_code_bytes(header)),
          one_bit_factor_(header.n),
          ex_factor_(header.n),
          one_bit_words_per_vector_(mvr_index_file_format::one_bit_bytes_per_vector(header) /
                                    sizeof(uint64_t)),
          one_bit_bytes_per_vector_(mvr_index_file_format::one_bit_bytes_per_vector(header)),
          full_code_bytes_per_vector_(mvr_index_file_format::full_code_bytes_per_vector(header))
    {
    }

    SplitQuantizedOutputBuffer(const SplitQuantizedOutputBuffer&) = delete;
    SplitQuantizedOutputBuffer& operator=(const SplitQuantizedOutputBuffer&) = delete;

    void copy_batch(
        size_t start,
        size_t cur_batch,
        const uint64_t* one_bit_code,
        const uint8_t* full_code,
        const float* one_bit_factor,
        const float* ex_factor)
    {
        const size_t one_bit_bytes = cur_batch * one_bit_bytes_per_vector_;
        const size_t full_code_bytes = cur_batch * full_code_bytes_per_vector_;
        const size_t factor_bytes = cur_batch * sizeof(float);

        std::memcpy(
            one_bit_code_.data() + start * one_bit_words_per_vector_,
            one_bit_code,
            one_bit_bytes);
        std::memcpy(
            full_code_.data() + start * full_code_bytes_per_vector_,
            full_code,
            full_code_bytes);
        std::memcpy(
            one_bit_factor_.data() + start,
            one_bit_factor,
            factor_bytes);
        std::memcpy(
            ex_factor_.data() + start,
            ex_factor,
            factor_bytes);
    }

    void write_to_files(
        const std::string& doc_1bit_path,
        const std::string& doc_4bit_path,
        const mvr_index_file_format::Header& header,
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

        mvr_index_file_format::write_header(doc_1bit_output, header, doc_1bit_path);
        mvr_index_file_format::save_rotator(doc_1bit_output, rotator, doc_1bit_path);
        mvr_index_file_format::write_header(doc_4bit_output, header, doc_4bit_path);
        mvr_index_file_format::save_rotator(doc_4bit_output, rotator, doc_4bit_path);

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

   private:
    std::vector<uint64_t> one_bit_code_;
    std::vector<uint8_t> full_code_;
    std::vector<float> one_bit_factor_;
    std::vector<float> ex_factor_;
    size_t one_bit_words_per_vector_ = 0;
    size_t one_bit_bytes_per_vector_ = 0;
    size_t full_code_bytes_per_vector_ = 0;
};

class ClusteredStage1OutputWriter
{
   public:
    ClusteredStage1OutputWriter(
        const std::string& path,
        const clustered_stage1_file_format::Header& header,
        const Rotator<float>& rotator)
        : path_(path)
    {
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        if (!output.is_open())
        {
            throw std::runtime_error("Failed to open clustered sidecar output file: " + path_);
        }

        clustered_stage1_file_format::write_header(output, header, path_);
        if (clustered_stage1_file_format::has_embedded_rotator(header))
        {
            clustered_stage1_file_format::save_rotator(output, rotator, path_);
        }

        const auto prefix_pos = output.tellp();
        if (prefix_pos == std::streampos(-1))
        {
            throw std::runtime_error(
                "Failed to determine clustered sidecar header size for: " + path_);
        }

        layout_.prefix_bytes = static_cast<size_t>(prefix_pos);
        layout_.code_bytes_per_vector = header.code_bytes_per_vector;
        layout_.code_offset = layout_.prefix_bytes;
        layout_.factor_offset =
            layout_.code_offset + header.n_entries * header.code_bytes_per_vector;
        layout_.doc_id_offset = layout_.factor_offset + header.n_entries * sizeof(float);
        layout_.token_to_cluster_pos_offset =
            layout_.doc_id_offset + header.n_entries * sizeof(int);
        layout_.total_bytes =
            layout_.token_to_cluster_pos_offset + header.n_entries * sizeof(uint32_t);
        output.close();

        if (layout_.prefix_bytes != clustered_stage1_file_format::prefix_bytes(header))
        {
            throw std::runtime_error("Unexpected cluster_1bit prefix size");
        }

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
        factor_data_ =
            reinterpret_cast<float*>(static_cast<char*>(mapping_) + layout_.factor_offset);
        doc_id_data_ =
            reinterpret_cast<int*>(static_cast<char*>(mapping_) + layout_.doc_id_offset);
        token_to_cluster_pos_data_ = reinterpret_cast<uint32_t*>(
            static_cast<char*>(mapping_) + layout_.token_to_cluster_pos_offset);
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

struct ProgressState
{
    const char* step_label = "";
    const char* status_label = "";
    size_t total_items = 0;
    std::atomic<size_t> handled_items {0};
    std::atomic<size_t> next_percent_to_report {1};
    std::chrono::steady_clock::time_point start_time {};
    std::mutex output_mutex;
};

void report_progress(ProgressState& progress, size_t handled_now)
{
    if (progress.total_items == 0 || handled_now == 0)
    {
        return;
    }

    const size_t handled =
        progress.handled_items.fetch_add(handled_now, std::memory_order_relaxed) +
        handled_now;
    const size_t percent =
        std::min<size_t>(100, handled * 100 / progress.total_items);

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

            const auto now = std::chrono::steady_clock::now();
            const double elapsed = elapsed_ms(progress.start_time, now);

            std::cout << "[build_fast] " << progress.step_label
                      << " progress: " << percent << "% "
                      << progress.status_label << " (" << handled << "/"
                      << progress.total_items << ")";
            if (handled < progress.total_items && elapsed > 0.0)
            {
                const double remaining_ms =
                    elapsed *
                    (static_cast<double>(progress.total_items - handled) /
                     static_cast<double>(handled));
                std::cout << " elapsed " << format_elapsed(elapsed)
                          << ", ETA " << format_elapsed(remaining_ms);
            }
            std::cout << "." << std::endl;
            return;
        }
    }
}

std::uintmax_t file_size_or_zero(const std::string& path)
{
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec)
    {
        return 0;
    }
    const auto size = std::filesystem::file_size(path, ec);
    if (ec)
    {
        return 0;
    }
    return size;
}

void validate_output_sizes(
    const std::string& index_dir,
    const mvr_index_file_format::Header& header)
{
    clustered_stage1_file_format::Header cluster_header;
    cluster_header.n_entries = header.n;
    cluster_header.code_bytes_per_vector =
        mvr_index_file_format::one_bit_bytes_per_vector(header);
    cluster_header.source_dim = header.d;
    cluster_header.padded_dim = header.padded_dim;
    cluster_header.rotator_type = header.rotator_type;
    cluster_header.rotator_bytes = mvr_index_file_format::rotator_bytes(header);

    const auto expected_doc_1bit =
        mvr_index_file_format::prefix_bytes(header) +
        mvr_index_file_format::doc_1bit_payload_bytes(header);
    const auto expected_doc_4bit =
        mvr_index_file_format::prefix_bytes(header) +
        mvr_index_file_format::doc_4bit_payload_bytes(header);
    const auto expected_cluster =
        clustered_stage1_file_format::prefix_bytes(cluster_header) +
        header.n * cluster_header.code_bytes_per_vector +
        header.n * sizeof(float) +
        header.n * sizeof(int) +
        header.n * sizeof(uint32_t);

    const auto doc_1bit_path = gpu_index_layout::doc_1bit_path(index_dir);
    const auto doc_4bit_path = gpu_index_layout::doc_4bit_path(index_dir);
    const auto cluster_1bit_path = gpu_index_layout::cluster_1bit_path(index_dir);

    if (file_size_or_zero(doc_1bit_path) != expected_doc_1bit)
    {
        throw std::runtime_error("doc_1bit.bin size does not match expected layout");
    }
    if (file_size_or_zero(doc_4bit_path) != expected_doc_4bit)
    {
        throw std::runtime_error("doc_4bit.bin size does not match expected layout");
    }
    if (file_size_or_zero(cluster_1bit_path) != expected_cluster)
    {
        throw std::runtime_error("cluster_1bit.bin size does not match expected layout");
    }
}

void write_optional_u64_json(
    std::ofstream& output,
    const char* key,
    const std::optional<std::uint64_t>& value,
    bool trailing_comma)
{
    output << "  \"" << key << "\": ";
    if (value.has_value())
    {
        output << *value;
    }
    else
    {
        output << "null";
    }
    if (trailing_comma)
    {
        output << ",";
    }
    output << "\n";
}

void write_index_metadata_json(
    const std::string& index_dir,
    const mvr_index_file_format::Header& header,
    const BuildMetadataOptions& options,
    int max_cluster_size_tokens)
{
    validate_output_sizes(index_dir, header);

    clustered_stage1_file_format::Header cluster_header;
    cluster_header.n_entries = header.n;
    cluster_header.code_bytes_per_vector =
        mvr_index_file_format::one_bit_bytes_per_vector(header);
    cluster_header.source_dim = header.d;
    cluster_header.padded_dim = header.padded_dim;
    cluster_header.rotator_type = header.rotator_type;
    cluster_header.rotator_bytes = mvr_index_file_format::rotator_bytes(header);

    const auto path = gpu_index_layout::metadata_path(index_dir);
    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open())
    {
        throw std::runtime_error("Failed to open metadata output file: " + path);
    }

    const auto doc_1bit_path = gpu_index_layout::doc_1bit_path(index_dir);
    const auto doc_4bit_path = gpu_index_layout::doc_4bit_path(index_dir);
    const auto doc_4bit_ex_path = gpu_index_layout::doc_4bit_ex_path(index_dir);
    const auto cluster_1bit_path = gpu_index_layout::cluster_1bit_path(index_dir);
    const auto ivf_path = gpu_index_layout::ivf_path(index_dir);
    const auto centroids_path = gpu_index_layout::centroids_path(index_dir);
    const auto centroids_hnsw_path = gpu_index_layout::centroids_hnsw_path(index_dir);
    const auto doclens_path = gpu_index_layout::doclens_path(index_dir);
    auto ex_sidecar_header = header;
    ex_sidecar_header.ex_bits = 4;

    const auto one_bit_code_bytes = mvr_index_file_format::one_bit_code_bytes(header);
    const auto one_bit_factor_bytes = mvr_index_file_format::one_bit_factor_bytes(header);
    const auto ex_factor_bytes = mvr_index_file_format::ex_factor_bytes(header);
    const auto full_code_bytes = mvr_index_file_format::full_code_bytes(header);
    const auto legacy_ex_code_bytes = mvr_index_file_format::legacy_ex_code_bytes(ex_sidecar_header);
    const auto doc_prefix_bytes = mvr_index_file_format::prefix_bytes(header);
    const auto doc_4bit_ex_prefix_bytes = mvr_index_file_format::prefix_bytes(ex_sidecar_header);
    const auto cluster_prefix_bytes = clustered_stage1_file_format::prefix_bytes(cluster_header);
    const auto cluster_code_bytes = header.n * cluster_header.code_bytes_per_vector;
    const auto cluster_factor_bytes = header.n * sizeof(float);
    const auto cluster_doc_id_bytes = header.n * sizeof(int);
    const auto cluster_token_to_cluster_pos_bytes = header.n * sizeof(uint32_t);

    output
        << "{\n"
        << "  \"format\": \"gpu_mvr_split_index_v2\",\n"
        << "  \"build_status\": \"complete\",\n"
        << "  \"builder\": \"" << options.builder_name << "\",\n"
        << "  \"built_at_utc\": \"" << current_utc_timestamp() << "\",\n"
        << "  \"clustered_only_rebuild\": "
        << (options.clustered_only_rebuild ? "true" : "false") << ",\n";
    write_optional_u64_json(output, "centroid_sample_seed", options.centroid_sample_seed, true);
    write_optional_u64_json(output, "rotator_seed", options.rotator_seed, true);
    output
        << "  \"embedding_scalar_type\": \"float32\",\n"
        << "  \"doc_id_scalar_type\": \"int32\",\n"
        << "  \"cluster_pos_scalar_type\": \"uint32\",\n"
        << "  \"binary_code_word_type\": \"uint64\",\n"
        << "  \"n\": " << header.n << ",\n"
        << "  \"d\": " << header.d << ",\n"
        << "  \"n_clusters\": " << header.n_clusters << ",\n"
        << "  \"ex_bits\": " << header.ex_bits << ",\n"
        << "  \"padded_dim\": " << header.padded_dim << ",\n"
        << "  \"max_cluster_size_tokens\": " << max_cluster_size_tokens << ",\n"
        << "  \"raw_embedding_bytes\": " << (header.n * header.d * sizeof(float)) << ",\n"
        << "  \"rotator_type\": \"" << mvr_index_file_format::rotator_type_name(header.rotator_type) << "\",\n"
        << "  \"rotator_serialized_bytes\": " << mvr_index_file_format::rotator_bytes(header) << ",\n"
        << "  \"files\": {\n"
        << "    \"doclens\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kDoclensFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(doclens_path) << ",\n"
        << "      \"content\": \"int32 document lengths, one entry per document\"\n"
        << "    },\n"
        << "    \"doc_1bit\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kDoc1BitFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(doc_1bit_path) << ",\n"
        << "      \"format\": \"mvr_index_file_format v1\",\n"
        << "      \"content\": \"original-order stage-1 payload: embedded rotator, 1-bit codes, one_bit_factor, ex_factor\",\n"
        << "      \"prefix_bytes\": " << doc_prefix_bytes << ",\n"
        << "      \"one_bit_code_offset_bytes\": " << doc_prefix_bytes << ",\n"
        << "      \"one_bit_code_bytes\": " << one_bit_code_bytes << ",\n"
        << "      \"one_bit_factor_offset_bytes\": " << (doc_prefix_bytes + one_bit_code_bytes) << ",\n"
        << "      \"one_bit_factor_bytes\": " << one_bit_factor_bytes << ",\n"
        << "      \"ex_factor_offset_bytes\": " << (doc_prefix_bytes + one_bit_code_bytes + one_bit_factor_bytes) << ",\n"
        << "      \"ex_factor_bytes\": " << ex_factor_bytes << "\n"
        << "    },\n"
        << "    \"doc_4bit\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kDoc4BitFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(doc_4bit_path) << ",\n"
        << "      \"format\": \"mvr_index_file_format v1\",\n"
        << "      \"content\": \"original-order stage-2 payload: embedded rotator and 4-bit quantized codes for each token embedding dimension\",\n"
        << "      \"prefix_bytes\": " << doc_prefix_bytes << ",\n"
        << "      \"full_code_offset_bytes\": " << doc_prefix_bytes << ",\n"
        << "      \"full_code_bytes\": " << full_code_bytes << "\n"
        << "    },\n"
        << "    \"doc_4bit_ex\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kDoc4BitExFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(doc_4bit_ex_path) << ",\n"
        << "      \"prefix_bytes\": " << doc_4bit_ex_prefix_bytes << ",\n"
        << "      \"ex_bits\": 4,\n"
        << "      \"ex_code_offset_bytes\": " << doc_4bit_ex_prefix_bytes << ",\n"
        << "      \"ex_code_bytes\": " << legacy_ex_code_bytes << ",\n"
        << "      \"ex_factor_offset_bytes\": " << (doc_4bit_ex_prefix_bytes + legacy_ex_code_bytes) << ",\n"
        << "      \"ex_factor_bytes\": " << mvr_index_file_format::ex_factor_bytes(ex_sidecar_header) << ",\n"
        << "      \"content\": \"original-order residual ex-bit sidecar for CPU reranking\"\n"
        << "    },\n"
        << "    \"cluster_1bit\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kCluster1BitFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(cluster_1bit_path) << ",\n"
        << "      \"format\": \"clustered_stage1_file_format v" << clustered_stage1_file_format::kVersion << "\",\n"
        << "      \"content\": \"cluster-ordered stage-1 sidecar: embedded rotator, 1-bit codes, one_bit_factor, doc_ids, token_to_cluster_pos\",\n"
        << "      \"prefix_bytes\": " << cluster_prefix_bytes << ",\n"
        << "      \"one_bit_code_offset_bytes\": " << cluster_prefix_bytes << ",\n"
        << "      \"one_bit_code_bytes\": " << cluster_code_bytes << ",\n"
        << "      \"one_bit_factor_offset_bytes\": " << (cluster_prefix_bytes + cluster_code_bytes) << ",\n"
        << "      \"one_bit_factor_bytes\": " << cluster_factor_bytes << ",\n"
        << "      \"doc_id_offset_bytes\": " << (cluster_prefix_bytes + cluster_code_bytes + cluster_factor_bytes) << ",\n"
        << "      \"doc_id_bytes\": " << cluster_doc_id_bytes << ",\n"
        << "      \"token_to_cluster_pos_offset_bytes\": "
        << (cluster_prefix_bytes + cluster_code_bytes + cluster_factor_bytes + cluster_doc_id_bytes) << ",\n"
        << "      \"token_to_cluster_pos_bytes\": " << cluster_token_to_cluster_pos_bytes << "\n"
        << "    },\n"
        << "    \"ivf\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kIvfFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(ivf_path) << ",\n"
        << "      \"content\": \"IVF posting lists and cluster metadata\"\n"
        << "    },\n"
        << "    \"centroids\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kCentroidsFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(centroids_path) << ",\n"
        << "      \"content\": \"persisted rotated centroid graph backing IVF_PG\"\n"
        << "    },\n"
        << "    \"centroids_hnsw\": {\n"
        << "      \"filename\": \"" << gpu_index_layout::kCentroidsHnswFilename << "\",\n"
        << "      \"size_bytes\": " << file_size_or_zero(centroids_hnsw_path) << ",\n"
        << "      \"content\": \"hnswlib HNSW index for the centroid set when available\"\n"
        << "    }\n"
        << "  }\n"
        << "}\n";
    if (!output)
    {
        throw std::runtime_error("Failed to write metadata output file: " + path);
    }
}

void copy_split_quantized_layout_fast(
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

void assign_and_quantize_shard_on_device(
    const MmapedEmbeddings& data,
    const std::vector<float>& centroids,
    size_t n_clusters,
    size_t assign_batch,
    int device_id,
    size_t quant_threads_per_worker,
    const Rotator<float>& rotator,
    size_t ex_bits,
    std::atomic<size_t>& next_batch_start,
    std::atomic<bool>& should_stop,
    std::vector<uint32_t>& list_nos,
    SplitQuantizedOutputBuffer& quantized_buffer,
    ProgressState& progress)
{
    BUILD_FAST_CUDA_CHECK(cudaSetDevice(device_id));

    struct BufferSlot
    {
        cudaStream_t stream = nullptr;
        float* d_q_ptr = nullptr;
        uint32_t* d_lab_ptr = nullptr;
        float* d_dist_ptr = nullptr;
        float* h_q_ptr = nullptr;
        uint32_t* h_labels = nullptr;
    };

    BufferSlot buffer;

    auto cleanup = [&]()
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
    };

    try
    {
        PG_CAGRA assignment_cagra(n_clusters, data.d);
        assignment_cagra.build_index(centroids.data());

        BUILD_FAST_CUDA_CHECK(cudaStreamCreate(&buffer.stream));
        BUILD_FAST_CUDA_CHECK(cudaMalloc(&buffer.d_q_ptr, assign_batch * data.d * sizeof(float)));
        BUILD_FAST_CUDA_CHECK(cudaMalloc(&buffer.d_lab_ptr, assign_batch * sizeof(uint32_t)));
        BUILD_FAST_CUDA_CHECK(cudaMalloc(&buffer.d_dist_ptr, assign_batch * sizeof(float)));
        BUILD_FAST_CUDA_CHECK(
            cudaMallocHost(&buffer.h_q_ptr, assign_batch * data.d * sizeof(float)));
        BUILD_FAST_CUDA_CHECK(cudaMallocHost(&buffer.h_labels, assign_batch * sizeof(uint32_t)));

        std::vector<float> rotated(assign_batch * PADDED_DIM);
        std::vector<uint64_t> one_bit_code_batch(assign_batch * PADDED_DIM / 64);
        std::vector<uint8_t> full_code_batch(assign_batch * PADDED_DIM * (1 + ex_bits) / 8);
        std::vector<float> one_bit_factor_batch(assign_batch);
        std::vector<float> ex_factor_batch(assign_batch);

        while (!should_stop.load(std::memory_order_relaxed))
        {
            const size_t start =
                next_batch_start.fetch_add(assign_batch, std::memory_order_relaxed);
            if (start >= data.num_embeddings)
            {
                break;
            }

            const size_t cur = std::min(assign_batch, data.num_embeddings - start);
            data.prefetch_embeddings(start, cur);
            data.copy_embeddings(start, cur, buffer.h_q_ptr);

            BUILD_FAST_CUDA_CHECK(cudaMemcpyAsync(
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

            BUILD_FAST_CUDA_CHECK(cudaMemcpyAsync(
                buffer.h_labels,
                buffer.d_lab_ptr,
                cur * sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                buffer.stream));

#pragma omp parallel for num_threads(quant_threads_per_worker)
            for (size_t i = 0; i < cur; ++i)
            {
                const float* raw_vec = buffer.h_q_ptr + i * data.d;
                rotator.rotate(raw_vec, &rotated[i * PADDED_DIM]);

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
                cur,
                one_bit_code_batch.data(),
                full_code_batch.data(),
                one_bit_factor_batch.data(),
                ex_factor_batch.data());

            BUILD_FAST_CUDA_CHECK(cudaStreamSynchronize(buffer.stream));
            std::memcpy(
                list_nos.data() + start,
                buffer.h_labels,
                cur * sizeof(uint32_t));
            report_progress(progress, cur);
        }
    }
    catch (...)
    {
        cleanup();
        throw;
    }

    cleanup();
}

std::vector<uint32_t> assign_and_quantize_multi_gpu(
    const MmapedEmbeddings& data,
    const std::vector<float>& centroids,
    size_t n_clusters,
    size_t assign_batch,
    const Rotator<float>& rotator,
    size_t ex_bits,
    SplitQuantizedOutputBuffer& quantized_buffer)
{
    auto device_ids = visible_gpu_ids();
    const size_t total_batches = (data.num_embeddings + assign_batch - 1) / assign_batch;
    const size_t worker_count = std::min(device_ids.size(), total_batches);
    device_ids.resize(worker_count);

    std::vector<uint32_t> list_nos(data.num_embeddings);
    if (worker_count == 0)
    {
        return list_nos;
    }

    omp_set_dynamic(0);
    const int omp_threads = std::max(1, omp_get_max_threads());
    const size_t quant_threads_per_worker =
        std::max<size_t>(1, static_cast<size_t>(omp_threads) / worker_count);

    std::cout << "[build_fast] Step 2: Assigning and quantizing embeddings with "
              << worker_count << " visible GPU(s) and up to "
              << quant_threads_per_worker << " CPU quantization threads per worker."
              << std::endl;

    std::atomic<size_t> next_batch_start {0};
    std::atomic<bool> should_stop {false};
    ProgressState progress;
    progress.step_label = "Step 2";
    progress.status_label = "embeddings assigned + quantized";
    progress.total_items = data.num_embeddings;
    progress.start_time = std::chrono::steady_clock::now();
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
                assign_and_quantize_shard_on_device(
                    data,
                    centroids,
                    n_clusters,
                    assign_batch,
                    device_id,
                    quant_threads_per_worker,
                    rotator,
                    ex_bits,
                    next_batch_start,
                    should_stop,
                    list_nos,
                    quantized_buffer,
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

    BUILD_FAST_CUDA_CHECK(cudaSetDevice(device_ids.front()));
    return list_nos;
}

}  // namespace

void build_index_fast(
    const MmapedEmbeddings& data,
    size_t n_clusters,
    size_t ex_bits,
    const std::vector<int>& doc_lens,
    const std::string& index_dir)
{
    using Clock = std::chrono::steady_clock;

    const size_t n = data.num_embeddings;
    const size_t d = data.d;
    const auto build_start = Clock::now();

    validate_doc_lens(doc_lens, n);

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
    const std::string centroids_path = gpu_index_layout::centroids_path(index_dir);

    data.advise_sequential();

    std::vector<float> centroids;

    std::cout << "[build_fast] Step 1: Randomly sampling " << n_clusters
              << " centroids from " << n << " vectors (d=" << d << ") ..." << std::endl;
    const auto step1_start = Clock::now();

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 centroid_rng(static_cast<std::mt19937::result_type>(kDefaultCentroidSampleSeed));
    std::shuffle(indices.begin(), indices.end(), centroid_rng);
    indices.resize(n_clusters);

    centroids.resize(n_clusters * d);
    for (size_t i = 0; i < n_clusters; ++i)
    {
        std::memcpy(&centroids[i * d], data.embedding_ptr(indices[i]), d * sizeof(float));
    }

    const auto step1_end = Clock::now();
    std::cout << "[build_fast] Step 1 done in "
              << format_elapsed(elapsed_ms(step1_start, step1_end))
              << ". Sampled centroids." << std::endl;

    constexpr RotatorType kRotatorType = RotatorType::FhtKacRotator;
    BUILD_FAST_CUDA_CHECK(cudaSetDevice(0));
    auto rotator = std::unique_ptr<Rotator<float>>(
        choose_rotator_with_seed<float>(
            d,
            kRotatorType,
            PADDED_DIM,
            kDefaultRotatorSeed));

    mvr_index_file_format::Header header;
    header.n = n;
    header.d = d;
    header.n_clusters = n_clusters;
    header.ex_bits = ex_bits;
    header.padded_dim = PADDED_DIM;
    header.rotator_type = kRotatorType;

    SplitQuantizedOutputBuffer quantized_buffer(header);

    constexpr size_t assign_batch = 65536;
    const auto step2_start = Clock::now();
    auto list_nos = assign_and_quantize_multi_gpu(
        data,
        centroids,
        n_clusters,
        assign_batch,
        *rotator,
        ex_bits,
        quantized_buffer);
    const auto step2_end = Clock::now();
    std::cout << "[build_fast] Step 2 done in "
              << format_elapsed(elapsed_ms(step2_start, step2_end))
              << ". Embeddings assigned and quantized in a single raw-data pass."
              << std::endl;

    std::cout << "[build_fast] Step 3: Rotating centroids and building persisted CAGRA graph on "
              << n_clusters << " centroids ..." << std::endl;
    const auto step3_start = Clock::now();

    std::vector<float> rotated_centroids(n_clusters * PADDED_DIM);
    for (size_t i = 0; i < n_clusters; ++i)
    {
        rotator->rotate(&centroids[i * d], &rotated_centroids[i * PADDED_DIM]);
    }

    auto pg_cagra = std::make_unique<PG_CAGRA>(n_clusters, PADDED_DIM);
    pg_cagra->build_index(rotated_centroids.data());

    const auto step3_end = Clock::now();
    std::cout << "[build_fast] Step 3 done in "
              << format_elapsed(elapsed_ms(step3_start, step3_end))
              << ". Persisted rotated CAGRA graph built." << std::endl;

    std::cout << "[build_fast] Step 4: Assembling IVF_PG ..." << std::endl;
    const auto step4_start = Clock::now();

    auto ivf = std::make_unique<IVF_PG>(n_clusters, d, PGType::CAGRA);
    delete ivf->pg_index;
    ivf->pg_index = pg_cagra.release();
    ivf->build_from_assignments(list_nos.data(), n);
    ivf->save(ivf_path, centroids_path);
    const int max_cluster_size_tokens = ivf->max_cluster_size();
    centroids.clear();
    centroids.shrink_to_fit();
    rotated_centroids.clear();
    rotated_centroids.shrink_to_fit();

    const auto step4_end = Clock::now();
    std::cout << "[build_fast] Step 4 done in "
              << format_elapsed(elapsed_ms(step4_start, step4_end))
              << ". IVF_PG assembled and saved. max_cluster_size_tokens="
              << max_cluster_size_tokens << "." << std::endl;

    std::cout << "[build_fast] Step 5: Writing quantized payloads to disk ..." << std::endl;
    const auto step5_start = Clock::now();
    quantized_buffer.write_to_files(
        doc_1bit_path,
        doc_4bit_path,
        header,
        *rotator);
    const auto step5_end = Clock::now();
    std::cout << "[build_fast] Step 5 done in "
              << format_elapsed(elapsed_ms(step5_start, step5_end))
              << ". doc_1bit.bin and doc_4bit.bin written in one shot." << std::endl;

    std::cout << "[build_fast] Step 6: Generating cluster_1bit.bin from doc_1bit.bin ..."
              << std::endl;
    const auto step6_start = Clock::now();
    BuildMetadataOptions metadata_options;
    metadata_options.builder_name = "gpu_build_fast";
    metadata_options.centroid_sample_seed = kDefaultCentroidSampleSeed;
    metadata_options.rotator_seed = kDefaultRotatorSeed;
    build_clustered_stage1_sidecar_fast(doc_lens, index_dir, index_dir);
    write_index_metadata_json(index_dir, header, metadata_options, max_cluster_size_tokens);
    const auto step6_end = Clock::now();
    std::cout << "[build_fast] Step 6 done in "
              << format_elapsed(elapsed_ms(step6_start, step6_end))
              << ". Clustered stage-1 payload and metadata saved."
              << std::endl;

    const auto build_end = Clock::now();
    std::cout << "[build_fast][profile] Total build time: "
              << format_elapsed(elapsed_ms(build_start, build_end)) << std::endl;
    std::cout << "[build_fast] Done. Index saved under: " << index_dir << std::endl;
}

void build_clustered_stage1_sidecar_fast(
    const std::vector<int>& doc_lens,
    const std::string& source_index_dir,
    const std::string& index_dir)
{
    using Clock = std::chrono::steady_clock;

    const auto build_start = Clock::now();
    std::filesystem::create_directories(index_dir);
    bool in_place_rebuild = false;
    std::error_code equivalent_ec;
    if (std::filesystem::exists(source_index_dir) &&
        std::filesystem::exists(index_dir))
    {
        in_place_rebuild =
            std::filesystem::equivalent(source_index_dir, index_dir, equivalent_ec);
    }
    std::error_code cleanup_ec;
    if (!in_place_rebuild)
    {
        std::filesystem::remove(
            gpu_index_layout::doc_1bit_path(index_dir), cleanup_ec);
        cleanup_ec.clear();
        std::filesystem::remove(
            gpu_index_layout::doc_4bit_path(index_dir), cleanup_ec);
        cleanup_ec.clear();
        std::filesystem::remove(
            gpu_index_layout::doc_4bit_ex_path(index_dir), cleanup_ec);
        cleanup_ec.clear();
        std::filesystem::remove(
            gpu_index_layout::metadata_path(index_dir), cleanup_ec);
        cleanup_ec.clear();
        std::filesystem::remove(
            gpu_index_layout::cpu_index_path(index_dir), cleanup_ec);
        cleanup_ec.clear();
        std::filesystem::remove(
            gpu_index_layout::centroids_hnsw_path(index_dir), cleanup_ec);
    }
    std::filesystem::remove(
        gpu_index_layout::cluster_1bit_path(index_dir), cleanup_ec);
    cleanup_ec.clear();
    std::filesystem::remove(
        gpu_index_layout::gpu_index_path(index_dir), cleanup_ec);

    const auto source_paths = gpu_index_layout::resolve_index_paths(source_index_dir);
    const auto source_ivf_path = gpu_index_layout::ivf_path(source_index_dir);
    const auto source_centroids_path = gpu_index_layout::centroids_path(source_index_dir);

    const auto target_doc_1bit_path = gpu_index_layout::doc_1bit_path(index_dir);
    const auto target_doc_4bit_path = gpu_index_layout::doc_4bit_path(index_dir);
    const auto target_doc_4bit_ex_path = gpu_index_layout::doc_4bit_ex_path(index_dir);
    const auto target_ivf_path = gpu_index_layout::ivf_path(index_dir);
    const auto target_centroids_path = gpu_index_layout::centroids_path(index_dir);
    const auto target_centroids_hnsw_path = gpu_index_layout::centroids_hnsw_path(index_dir);
    const auto target_clustered_stage1_path = gpu_index_layout::cluster_1bit_path(index_dir);

    std::cout << "[build_fast] Sidecar mode: cloning base index from "
              << source_index_dir << " to " << index_dir << std::endl;

    const auto copy_start = Clock::now();
    copy_split_quantized_layout_fast(
        source_paths,
        target_doc_1bit_path,
        target_doc_4bit_path,
        target_doc_4bit_ex_path,
        target_centroids_hnsw_path);
    if (source_ivf_path != target_ivf_path)
    {
        std::filesystem::copy_file(
            source_ivf_path,
            target_ivf_path,
            std::filesystem::copy_options::overwrite_existing);
    }
    if (source_centroids_path != target_centroids_path)
    {
        std::filesystem::copy_file(
            source_centroids_path,
            target_centroids_path,
            std::filesystem::copy_options::overwrite_existing);
    }
    std::cout << "[build_fast] Base index cloned in "
              << format_elapsed(elapsed_ms(copy_start, Clock::now())) << std::endl;

    std::ifstream inf(target_doc_1bit_path, std::ios::binary);
    const auto header = mvr_index_file_format::read_header(inf, target_doc_1bit_path);
    const size_t n = header.n;
    const size_t code_bytes_per_vector = mvr_index_file_format::one_bit_bytes_per_vector(header);
    if (header.padded_dim != PADDED_DIM)
    {
        throw std::runtime_error(
            "Index file padded_dim=" + std::to_string(header.padded_dim) +
            " does not match compiled PADDED_DIM=" + std::to_string(PADDED_DIM));
    }

    auto rotator = std::unique_ptr<Rotator<float>>(
        mvr_index_file_format::load_rotator(inf, header, target_doc_1bit_path));
    const auto prefix_bytes = static_cast<size_t>(inf.tellg());
    inf.close();

    validate_doc_lens(doc_lens, n);
    const auto doc_ptrs = build_doc_ptrs(doc_lens);

    IVF_PG ivf(header.n_clusters, header.d, PGType::CAGRA);
    ivf.load(target_ivf_path, target_centroids_path);
    const int max_cluster_size_tokens = ivf.max_cluster_size();
    if (ivf.inv_list.size() != n)
    {
        throw std::runtime_error(
            "IVF posting count " + std::to_string(ivf.inv_list.size()) +
            " does not match token count " + std::to_string(n));
    }

    std::vector<uint32_t> cluster_rank(n);
    for (size_t pos = 0; pos < ivf.inv_list.size(); ++pos)
    {
        cluster_rank[ivf.inv_list[pos]] = static_cast<uint32_t>(pos);
    }

    clustered_stage1_file_format::Header cluster_header;
    cluster_header.n_entries = ivf.inv_list.size();
    cluster_header.code_bytes_per_vector = code_bytes_per_vector;
    cluster_header.source_dim = header.d;
    cluster_header.padded_dim = header.padded_dim;
    cluster_header.rotator_type = header.rotator_type;
    cluster_header.rotator_bytes = mvr_index_file_format::rotator_bytes(header);

    ClusteredStage1OutputWriter clustered_stage1_writer(
        target_clustered_stage1_path,
        cluster_header,
        *rotator);

    std::ifstream code_file(target_doc_1bit_path, std::ios::binary);
    std::ifstream factor_file(target_doc_1bit_path, std::ios::binary);
    code_file.seekg(static_cast<std::streamoff>(prefix_bytes));
    factor_file.seekg(static_cast<std::streamoff>(
        prefix_bytes + mvr_index_file_format::one_bit_code_bytes(header)));
    if (!code_file || !factor_file)
    {
        throw std::runtime_error(
            "Failed to position doc_1bit.bin while generating cluster_1bit.bin");
    }

    const auto sidecar_start = Clock::now();
    const size_t batch_vectors = 1 << 20;
    std::vector<uint64_t> one_bit_code_batch(batch_vectors * PADDED_DIM / 64);
    std::vector<float> one_bit_factor_batch(batch_vectors);

    for (size_t start = 0; start < n; start += batch_vectors)
    {
        const size_t cur_batch = std::min(batch_vectors, n - start);
        const size_t cur_code_bytes = cur_batch * code_bytes_per_vector;
        const size_t cur_factor_bytes = cur_batch * sizeof(float);

        code_file.read(
            reinterpret_cast<char*>(one_bit_code_batch.data()),
            static_cast<std::streamsize>(cur_code_bytes));
        factor_file.read(
            reinterpret_cast<char*>(one_bit_factor_batch.data()),
            static_cast<std::streamsize>(cur_factor_bytes));
        if (!code_file || !factor_file)
        {
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

    BuildMetadataOptions metadata_options;
    metadata_options.builder_name = "gpu_build_fast";
    metadata_options.clustered_only_rebuild = true;
    if (rotator->has_seed())
    {
        metadata_options.rotator_seed = rotator->seed();
    }
    write_index_metadata_json(index_dir, header, metadata_options, max_cluster_size_tokens);

    std::cout << "[build_fast] cluster_1bit.bin generated in "
              << format_elapsed(elapsed_ms(sidecar_start, Clock::now())) << std::endl;
    std::cout << "[build_fast][profile] Fast sidecar build total: "
              << format_elapsed(elapsed_ms(build_start, Clock::now())) << std::endl;
}
