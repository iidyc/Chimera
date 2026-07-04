#include "build_index/build_utils.cuh"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace Chimera {

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

            std::cout << "[build_index] " << progress.step_label
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

}  // namespace Chimera
