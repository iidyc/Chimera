#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#include "rabitqlib/utils/rotator.hpp"

namespace Chimera {

namespace clustered_format {

inline constexpr char kMagic[] = {'M', 'V', 'R', 'C', 'L', 'S', 'T', '1'};
inline constexpr std::uint32_t kVersion = 3;

struct Header {
    std::uint32_t version = kVersion;
    size_t n_entries = 0;
    size_t code_bytes_per_vector = 0;
    size_t source_dim = 0;
    size_t padded_dim = 0;
    rabitqlib::RotatorType rotator_type = rabitqlib::RotatorType::FhtKacRotator;
    size_t rotator_bytes = 0;
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

    write_value(output, header.version, "version", filename);
    write_value(output, header.n_entries, "n_entries", filename);
    write_value(output, header.code_bytes_per_vector, "code_bytes_per_vector", filename);
    write_value(output, header.source_dim, "source_dim", filename);
    write_value(output, header.padded_dim, "padded_dim", filename);
    write_value(output, header.rotator_type, "rotator_type", filename);
    write_value(output, header.rotator_bytes, "rotator_bytes", filename);
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
    Header header;
    header.version = version;
    if (version == 2) {
        header.n_entries = read_value<size_t>(input, "n_entries", filename);
        header.code_bytes_per_vector =
            read_value<size_t>(input, "code_bytes_per_vector", filename);
        return header;
    }
    if (version != kVersion) {
        throw std::runtime_error(
            "Unsupported clustered sidecar version " + std::to_string(version) + " for " + filename
        );
    }

    header.n_entries = read_value<size_t>(input, "n_entries", filename);
    header.code_bytes_per_vector =
        read_value<size_t>(input, "code_bytes_per_vector", filename);
    header.source_dim = read_value<size_t>(input, "source_dim", filename);
    header.padded_dim = read_value<size_t>(input, "padded_dim", filename);
    header.rotator_type =
        read_value<rabitqlib::RotatorType>(input, "rotator_type", filename);
    header.rotator_bytes = read_value<size_t>(input, "rotator_bytes", filename);
    return header;
}

inline size_t header_bytes() {
    return sizeof(kMagic) +
           sizeof(std::uint32_t) +
           5 * sizeof(size_t) +
           sizeof(rabitqlib::RotatorType);
}

inline size_t header_bytes(std::uint32_t version) {
    if (version == 2) {
        return sizeof(kMagic) + sizeof(std::uint32_t) + 2 * sizeof(size_t);
    }
    return header_bytes();
}

inline size_t header_bytes(const Header& header) {
    return header_bytes(header.version);
}

inline bool has_embedded_rotator(const Header& header) {
    return header.rotator_bytes > 0;
}

inline size_t prefix_bytes(const Header& header) {
    return header_bytes(header) + header.rotator_bytes;
}

inline void save_rotator(
    std::ofstream& output,
    const rabitqlib::Rotator<float>& rotator,
    const std::string& filename
) {
    rotator.save(output);
    if (!output) {
        throw std::runtime_error(
            "Failed to write rotator to clustered sidecar: " + filename
        );
    }
}

inline rabitqlib::Rotator<float>* load_rotator(
    std::ifstream& input,
    const Header& header,
    const std::string& filename
) {
    if (!has_embedded_rotator(header)) {
        return nullptr;
    }
    auto* rotator = rabitqlib::choose_rotator<float>(
        header.source_dim,
        header.rotator_type,
        header.padded_dim
    );
    rotator->load(input);
    if (!input) {
        delete rotator;
        throw std::runtime_error(
            "Failed to read rotator from clustered sidecar: " + filename
        );
    }
    return rotator;
}

}  // namespace clustered_format


}  // namespace Chimera
