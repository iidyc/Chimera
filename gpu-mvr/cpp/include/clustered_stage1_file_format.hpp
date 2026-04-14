#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace clustered_stage1_file_format {

inline constexpr char kMagic[] = {'M', 'V', 'R', 'C', 'L', 'S', 'T', '1'};
inline constexpr std::uint32_t kVersion = 2;

struct Header {
    size_t n_entries = 0;
    size_t code_bytes_per_vector = 0;
};

template <typename T>
inline void write_value(
    std::ofstream& output,
    const T& value,
    const char* field_name,
    const std::string& filename
) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!output) {
        throw std::runtime_error(
            "Failed to write " + std::string(field_name) + " to clustered sidecar: " + filename
        );
    }
}

template <typename T>
inline T read_value(std::ifstream& input, const char* field_name, const std::string& filename) {
    T value {};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error(
            "Failed to read " + std::string(field_name) + " from clustered sidecar: " + filename
        );
    }
    return value;
}

inline void write_header(
    std::ofstream& output,
    const Header& header,
    const std::string& filename
) {
    output.write(kMagic, sizeof(kMagic));
    if (!output) {
        throw std::runtime_error("Failed to write clustered sidecar magic to: " + filename);
    }

    write_value(output, kVersion, "version", filename);
    write_value(output, header.n_entries, "n_entries", filename);
    write_value(output, header.code_bytes_per_vector, "code_bytes_per_vector", filename);
}

inline Header read_header(std::ifstream& input, const std::string& filename) {
    char magic[sizeof(kMagic)] = {};
    input.read(magic, sizeof(magic));
    if (!input) {
        throw std::runtime_error("Failed to read clustered sidecar header from: " + filename);
    }
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("Invalid clustered sidecar magic in: " + filename);
    }

    const auto version = read_value<std::uint32_t>(input, "version", filename);
    if (version != kVersion) {
        throw std::runtime_error(
            "Unsupported clustered sidecar version " + std::to_string(version) + " for " + filename
        );
    }

    Header header;
    header.n_entries = read_value<size_t>(input, "n_entries", filename);
    header.code_bytes_per_vector =
        read_value<size_t>(input, "code_bytes_per_vector", filename);
    return header;
}

inline size_t header_bytes() {
    return sizeof(kMagic) + sizeof(std::uint32_t) + 2 * sizeof(size_t);
}

}  // namespace clustered_stage1_file_format
