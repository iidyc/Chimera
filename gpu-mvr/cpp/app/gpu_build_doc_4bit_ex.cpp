#include "arg_utils.hpp"
#include "gpu_index_layout.hpp"
#include "io.hpp"
#include "mvr_index_file_format.hpp"
#include "quantization.hpp"

#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace {

struct ExSidecarLayout {
    size_t prefix_bytes = 0;
    size_t ex_code_offset = 0;
    size_t ex_factor_offset = 0;
    size_t ex_code_bytes_per_vector = 0;
    size_t total_bytes = 0;
};

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

    if (milliseconds >= kMsPerMinute) {
        out << (milliseconds / kMsPerMinute) << " minutes";
    } else {
        out << (milliseconds / kMsPerSecond) << " seconds";
    }
    return out.str();
}

void print_help(const char* program)
{
    std::cout
        << "Usage: " << program
        << " --index_dir <index_dir> --data <embeddings.bin> [--output <doc_4bit_ex.bin>] [--ex_bits <count>] [--batch_size <count>] [--threads <count>]\n\n"
        << "Arguments:\n"
        << "  --index_dir   Existing split index directory containing doc_1bit.bin.\n"
        << "  --data        Raw embeddings.bin used to build the index.\n"
        << "  --output      Output sidecar path. Default: <index_dir>/doc_4bit_ex.bin\n"
        << "  --ex_bits     Number of residual bits to encode. Default: 4\n"
        << "  --batch_size  Embeddings per quantization batch. Default: 16384\n"
        << "  --threads     OpenMP threads for CPU rotation/quantization. Default: current OMP max\n";
}

void write_all_at(
    int fd,
    const void* src,
    size_t bytes,
    size_t offset,
    const std::string& filename)
{
    const auto* ptr = static_cast<const char*>(src);
    size_t written_total = 0;
    while (written_total < bytes) {
        const auto cur_offset = static_cast<off_t>(offset + written_total);
        const auto cur_bytes = bytes - written_total;
        const ssize_t written = pwrite(fd, ptr + written_total, cur_bytes, cur_offset);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(
                "Failed to write doc_4bit_ex batch to " + filename + ": " +
                std::system_category().message(errno));
        }
        written_total += static_cast<size_t>(written);
    }
}

class ExSidecarWriter
{
   public:
    ExSidecarWriter(
        const std::string& path,
        const mvr_index_file_format::Header& header,
        const rabitqlib::Rotator<float>& rotator)
        : path_(path)
    {
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        if (!output.is_open()) {
            throw std::runtime_error("Failed to open output file: " + path_);
        }

        mvr_index_file_format::write_header(output, header, path_);
        mvr_index_file_format::save_rotator(output, rotator, path_);

        const auto prefix_pos = output.tellp();
        if (prefix_pos == std::streampos(-1)) {
            throw std::runtime_error("Failed to determine sidecar header size for: " + path_);
        }

        layout_.prefix_bytes = static_cast<size_t>(prefix_pos);
        layout_.ex_code_bytes_per_vector =
            mvr_index_file_format::legacy_ex_code_bytes_per_vector(header);
        layout_.ex_code_offset = layout_.prefix_bytes;
        layout_.ex_factor_offset =
            layout_.ex_code_offset + header.n * layout_.ex_code_bytes_per_vector;
        layout_.total_bytes = layout_.ex_factor_offset + header.n * sizeof(float);
        output.close();

        fd_ = open(path_.c_str(), O_WRONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to reopen output file: " + path_);
        }
        if (ftruncate(fd_, static_cast<off_t>(layout_.total_bytes)) != 0) {
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
        if (fd_ >= 0) {
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

}  // namespace

int main(int argc, char* argv[])
{
    std::string index_dir;
    std::string data_path;
    std::string output_path;
    size_t ex_bits = 4;
    size_t batch_size = 16384;
    int threads = 0;

    if (argc == 1) {
        print_help(argv[0]);
        return 1;
    }

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_help(argv[0]);
                return 0;
            }
            if (arg == "--index_dir") {
                index_dir = require_value(argc, argv, i, arg);
                continue;
            }
            if (arg == "--data") {
                data_path = require_value(argc, argv, i, arg);
                continue;
            }
            if (arg == "--output") {
                output_path = require_value(argc, argv, i, arg);
                continue;
            }
            if (arg == "--ex_bits") {
                ex_bits = std::stoull(require_value(argc, argv, i, arg));
                continue;
            }
            if (arg == "--batch_size") {
                batch_size = std::stoull(require_value(argc, argv, i, arg));
                continue;
            }
            if (arg == "--threads") {
                threads = std::stoi(require_value(argc, argv, i, arg));
                continue;
            }
            throw std::runtime_error("Unknown argument: " + arg);
        }
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        print_help(argv[0]);
        return 1;
    }

    if (index_dir.empty() || data_path.empty()) {
        std::cerr << "Missing required arguments.\n\n";
        print_help(argv[0]);
        return 1;
    }
    if (ex_bits == 0 || ex_bits > 8) {
        std::cerr << "--ex_bits must be in [1, 8]\n";
        return 1;
    }
    if (batch_size == 0) {
        std::cerr << "--batch_size must be > 0\n";
        return 1;
    }

    if (output_path.empty()) {
        output_path = gpu_index_layout::doc_4bit_ex_path(index_dir);
    }

    const auto start = std::chrono::steady_clock::now();
    const auto doc1_path = gpu_index_layout::doc_1bit_path(index_dir);
    std::ifstream doc1(doc1_path, std::ios::binary);
    if (!doc1.is_open()) {
        std::cerr << "Failed to open " << doc1_path << "\n";
        return 1;
    }

    auto header = mvr_index_file_format::read_header(doc1, doc1_path);
    std::unique_ptr<rabitqlib::Rotator<float>> rotator(
        mvr_index_file_format::load_rotator(doc1, header, doc1_path));
    doc1.close();

    auto data = load_data_mmap(data_path);
    if (data.num_embeddings != header.n || data.d != header.d) {
        std::cerr << "Raw embedding file does not match doc_1bit.bin metadata: "
                  << "data has n=" << data.num_embeddings << " d=" << data.d
                  << ", index expects n=" << header.n << " d=" << header.d << "\n";
        return 1;
    }

    if (threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }
    const int active_threads = omp_get_max_threads();

    header.ex_bits = ex_bits;
    ExSidecarWriter writer(output_path, header, *rotator);
    const size_t padded_dim = header.padded_dim;
    const size_t ex_code_stride = mvr_index_file_format::legacy_ex_code_bytes_per_vector(header);

    std::vector<float> raw_batch(batch_size * data.d);
    std::vector<float> rotated_batch(batch_size * padded_dim);
    std::vector<uint8_t> ex_code_batch(batch_size * ex_code_stride);
    std::vector<float> ex_factor_batch(batch_size);

    data.advise_sequential();

    const auto quant_start = std::chrono::steady_clock::now();
    auto last_progress = quant_start;
    for (size_t start_idx = 0; start_idx < data.num_embeddings; start_idx += batch_size) {
        const size_t count = std::min(batch_size, data.num_embeddings - start_idx);
        data.prefetch_embeddings(start_idx, count);
        data.copy_embeddings(start_idx, count, raw_batch.data());

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; ++i) {
            rotator->rotate(
                raw_batch.data() + i * data.d,
                rotated_batch.data() + i * padded_dim);
            encode_ex_bits(
                rotated_batch.data() + i * padded_dim,
                padded_dim,
                ex_bits,
                ex_code_batch.data() + i * ex_code_stride,
                &ex_factor_batch[i]);
        }

        writer.write_batch(
            start_idx,
            count,
            ex_code_batch.data(),
            ex_factor_batch.data());

        const auto now = std::chrono::steady_clock::now();
        if (now - last_progress >= std::chrono::seconds(5) ||
            start_idx + count == data.num_embeddings) {
            const size_t done = start_idx + count;
            const double seconds =
                std::chrono::duration<double>(now - quant_start).count();
            const double rate = seconds > 0.0 ? done / seconds : 0.0;
            const double pct = 100.0 * static_cast<double>(done) /
                               static_cast<double>(data.num_embeddings);
            std::cout << "[doc_4bit_ex] progress=" << std::fixed << std::setprecision(2)
                      << pct << "% (" << done << "/" << data.num_embeddings
                      << "), rate=" << std::setprecision(0) << rate
                      << " vectors/s, threads=" << active_threads << std::endl;
            last_progress = now;
        }
    }

    const auto end = std::chrono::steady_clock::now();
    std::cout << "[doc_4bit_ex] wrote " << output_path
              << " with ex_bits=" << ex_bits
              << " in " << format_elapsed(elapsed_ms(start, end)) << std::endl;
    return 0;
}
