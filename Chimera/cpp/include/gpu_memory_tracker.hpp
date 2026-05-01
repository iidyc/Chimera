#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>

namespace Chimera {

class GpuMemoryTracker {
public:
    void sample(const std::string& label) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (status != cudaSuccess) {
            return;
        }

        enabled_ = true;
        total_bytes_ = total_bytes;
        current_used_bytes_ = total_bytes - free_bytes;
        ++sample_count_;

        if (current_used_bytes_ >= peak_used_bytes_) {
            peak_used_bytes_ = current_used_bytes_;
            peak_label_ = label;
        }
    }

    bool should_sample_query(size_t query_index, size_t total_queries) const {
        if (total_queries <= 128) {
            return true;
        }
        if (query_index < 8) {
            return true;
        }
        if ((query_index + 1) % 64 == 0) {
            return true;
        }
        return query_index + 1 == total_queries;
    }

    void sample_query_if_needed(
        const std::string& phase,
        size_t query_index,
        size_t total_queries)
    {
        if (!should_sample_query(query_index, total_queries)) {
            return;
        }
        sample(phase + "_query_" + std::to_string(query_index));
    }

    void print_summary() const {
        if (!enabled_) {
            return;
        }

        const auto current_mib =
            static_cast<double>(current_used_bytes_) / (1024.0 * 1024.0);
        const auto peak_mib =
            static_cast<double>(peak_used_bytes_) / (1024.0 * 1024.0);
        const auto total_mib =
            static_cast<double>(total_bytes_) / (1024.0 * 1024.0);

        auto old_flags = std::cout.flags();
        auto old_precision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[GPU_MEM] current=" << current_mib
                  << " MiB, peak=" << peak_mib
                  << " MiB, total=" << total_mib
                  << " MiB, samples=" << sample_count_
                  << ", peak_label=" << peak_label_
                  << std::endl;
        std::cout.flags(old_flags);
        std::cout.precision(old_precision);
    }

private:
    bool enabled_ = false;
    size_t total_bytes_ = 0;
    size_t current_used_bytes_ = 0;
    size_t peak_used_bytes_ = 0;
    size_t sample_count_ = 0;
    std::string peak_label_;
};

}  // namespace Chimera
