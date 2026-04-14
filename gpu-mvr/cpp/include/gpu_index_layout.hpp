#pragma once

#include <filesystem>
#include <string>

namespace gpu_index_layout {

inline constexpr char kIvfFilename[] = "ivf.bin";
inline constexpr char kCpuIndexFilename[] = "cpu_index.bin";
inline constexpr char kLegacyQuantizedDataFilename[] = "quantized_data.bin";
inline constexpr char kCentroidsFilename[] = "centroids.carga";
inline constexpr char kGpuIndexFilename[] = "gpu_index.bin";
inline constexpr char kLegacyClusteredStage1Filename[] = "clustered_stage1.bin";

struct ResolvedPaths {
    bool split_layout = false;
    std::string quantized_data_path;
    std::string ivf_path;
    std::string centroids_path;
    std::string gpu_index_path;
};

inline std::string cpu_index_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kCpuIndexFilename).string();
}

inline std::string legacy_quantized_data_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kLegacyQuantizedDataFilename).string();
}

inline std::string ivf_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kIvfFilename).string();
}

inline std::string centroids_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kCentroidsFilename).string();
}

inline std::string gpu_index_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kGpuIndexFilename).string();
}

inline std::string legacy_clustered_stage1_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kLegacyClusteredStage1Filename).string();
}

inline std::string resolve_existing_cpu_index_path(const std::string& index_dir) {
    const auto cpu_path = cpu_index_path(index_dir);
    if (std::filesystem::exists(cpu_path)) {
        return cpu_path;
    }
    return legacy_quantized_data_path(index_dir);
}

inline std::string resolve_existing_gpu_index_path(const std::string& index_dir) {
    const auto gpu_path = gpu_index_path(index_dir);
    if (std::filesystem::exists(gpu_path)) {
        return gpu_path;
    }
    return legacy_clustered_stage1_path(index_dir);
}

inline ResolvedPaths resolve_index_paths(const std::string& path) {
    const std::filesystem::path fs_path(path);

    if (std::filesystem::is_directory(fs_path)) {
        const auto index_dir = path;
        return {
            true,
            resolve_existing_cpu_index_path(index_dir),
            ivf_path(index_dir),
            centroids_path(index_dir),
            resolve_existing_gpu_index_path(index_dir),
        };
    }

    if (fs_path.filename() == kCpuIndexFilename ||
        fs_path.filename() == kLegacyQuantizedDataFilename) {
        const auto index_dir = fs_path.parent_path().string();
        return {
            true,
            fs_path.string(),
            ivf_path(index_dir),
            centroids_path(index_dir),
            resolve_existing_gpu_index_path(index_dir),
        };
    }

    return {
        false,
        path,
        path,
        "",
        "",
    };
}

}  // namespace gpu_index_layout
