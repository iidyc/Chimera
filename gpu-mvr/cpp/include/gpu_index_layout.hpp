#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

namespace gpu_index_layout {

inline constexpr char kIvfFilename[] = "ivf.bin";
inline constexpr char kDoc1BitFilename[] = "doc_1bit.bin";
inline constexpr char kDoc4BitFilename[] = "doc_4bit.bin";
inline constexpr char kCluster1BitFilename[] = "cluster_1bit.bin";
inline constexpr char kDoclensFilename[] = "doclens.bin";
inline constexpr char kMetadataFilename[] = "index_metadata.json";
inline constexpr char kCpuIndexFilename[] = "cpu_index.bin";
inline constexpr char kCentroidsFilename[] = "centroids.carga";
inline constexpr char kGpuIndexFilename[] = "gpu_index.bin";

struct ResolvedPaths {
    std::string quantized_data_path;
    std::string doc_4bit_path;
    std::string ivf_path;
    std::string centroids_path;
    std::string gpu_index_path;
    std::string metadata_path;
};

inline std::string doc_1bit_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kDoc1BitFilename).string();
}

inline std::string doc_4bit_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kDoc4BitFilename).string();
}

inline std::string cluster_1bit_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kCluster1BitFilename).string();
}

inline std::string metadata_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kMetadataFilename).string();
}

inline std::string doclens_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kDoclensFilename).string();
}

inline std::string cpu_index_path(const std::string& index_dir) {
    return (std::filesystem::path(index_dir) / kCpuIndexFilename).string();
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

inline std::string index_dir_from_index_path(const std::string& path) {
    const std::filesystem::path fs_path(path);
    if (std::filesystem::is_directory(fs_path)) {
        return fs_path.string();
    }
    return fs_path.parent_path().string();
}

inline std::string doclens_path_for_index(const std::string& path) {
    return doclens_path(index_dir_from_index_path(path));
}

inline ResolvedPaths resolve_index_paths(const std::string& path) {
    const std::filesystem::path fs_path(path);

    if (std::filesystem::is_directory(fs_path)) {
        const auto index_dir = path;
        return {
            doc_1bit_path(index_dir),
            doc_4bit_path(index_dir),
            ivf_path(index_dir),
            centroids_path(index_dir),
            cluster_1bit_path(index_dir),
            metadata_path(index_dir),
        };
    }

    if (fs_path.filename() == kDoc1BitFilename) {
        const auto index_dir = fs_path.parent_path().string();
        return {
            fs_path.string(),
            doc_4bit_path(index_dir),
            ivf_path(index_dir),
            centroids_path(index_dir),
            cluster_1bit_path(index_dir),
            metadata_path(index_dir),
        };
    }

    throw std::runtime_error(
        "Expected --index to be an index directory or " + std::string(kDoc1BitFilename) +
        ", got: " + path);
}

}  // namespace gpu_index_layout
