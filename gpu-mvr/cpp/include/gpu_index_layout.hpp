#pragma once

#include <filesystem>
#include <string>

namespace gpu_index_layout {

inline constexpr char kIvfFilename[] = "ivf.bin";
inline constexpr char kQuantizedDataFilename[] = "quantized_data.bin";
inline constexpr char kCentroidsFilename[] = "centroids.carga";

struct ResolvedPaths {
    bool split_layout = false;
    std::string quantized_data_path;
    std::string ivf_path;
    std::string centroids_path;
};

inline std::string quantized_data_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kQuantizedDataFilename).string();
}

inline std::string ivf_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kIvfFilename).string();
}

inline std::string centroids_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kCentroidsFilename).string();
}

inline ResolvedPaths resolve_index_paths(const std::string& path) {
    const std::filesystem::path fs_path(path);

    if (std::filesystem::is_directory(fs_path)) {
        return {
            true,
            quantized_data_path(path),
            ivf_path(path),
            centroids_path(path),
        };
    }

    if (fs_path.filename() == kQuantizedDataFilename) {
        const auto index_dir = fs_path.parent_path().string();
        return {
            true,
            fs_path.string(),
            ivf_path(index_dir),
            centroids_path(index_dir),
        };
    }

    return {
        false,
        path,
        path,
        "",
    };
}

}  // namespace gpu_index_layout
