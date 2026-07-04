#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace Chimera {

void build_cuda_check(cudaError_t status, const char* expr, const char* file, int line);

#define BUILD_CUDA_CHECK(call) \
    ::Chimera::build_cuda_check((call), #call, __FILE__, __LINE__)

double elapsed_ms(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end);

std::string format_elapsed(double milliseconds);

std::string current_utc_timestamp();

std::uintmax_t file_size_or_zero(const std::string& path);

std::vector<int> visible_gpu_ids();

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

void report_progress(ProgressState& progress, size_t handled_now);

}  // namespace Chimera
