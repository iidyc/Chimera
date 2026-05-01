#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#include "rabitqlib/utils/rotator.hpp"

namespace Chimera {

namespace doc_format {

inline constexpr char kMagic[] = {'M', 'V', 'R', 'I', 'D', 'X', 'v', '1'};
inline constexpr std::uint32_t kVersion = 1;

struct Header {
    size_t n = 0;
    size_t d = 0;
    size_t n_clusters = 0;
    size_t ex_bits = 0;
    size_t padded_dim = 0;
    rabitqlib::RotatorType rotator_type = rabitqlib::RotatorType::FhtKacRotator;
};

inline const char* rotator_type_name(rabitqlib::RotatorType rotator_type) {
    switch (rotator_type) {
        case rabitqlib::RotatorType::MatrixRotator:
            return "MatrixRotator";
        case rabitqlib::RotatorType::FhtKacRotator:
            return "FhtKacRotator";
        default:
            return "Unknown";
    }
}

inline size_t one_bit_bytes_per_vector(const Header& header) {
    return header.padded_dim / 8;
}

inline size_t full_code_bytes_per_vector(const Header& header) {
    return header.padded_dim * (1 + header.ex_bits) / 8;
}

inline size_t legacy_ex_code_bytes_per_vector(const Header& header) {
    return header.padded_dim * header.ex_bits / 8;
}

inline size_t one_bit_code_bytes(const Header& header) {
    return header.n * one_bit_bytes_per_vector(header);
}

inline size_t full_code_bytes(const Header& header) {
    return header.n * full_code_bytes_per_vector(header);
}

inline size_t legacy_ex_code_bytes(const Header& header) {
    return header.n * legacy_ex_code_bytes_per_vector(header);
}

inline size_t one_bit_factor_bytes(const Header& header) {
    return header.n * sizeof(float);
}

inline size_t ex_factor_bytes(const Header& header) {
    return header.n * sizeof(float);
}

inline size_t header_bytes() {
    return sizeof(kMagic) +
           sizeof(std::uint32_t) +
           5 * sizeof(size_t) +
           sizeof(rabitqlib::RotatorType);
}

inline size_t rotator_bytes(const Header& header) {
    switch (header.rotator_type) {
        case rabitqlib::RotatorType::MatrixRotator:
            return header.d * header.padded_dim * sizeof(float);
        case rabitqlib::RotatorType::FhtKacRotator:
            return header.padded_dim / 2;
        default:
            throw std::runtime_error(
                "Unsupported rotator type while computing serialized size");
    }
}

inline size_t prefix_bytes(const Header& header) {
    return header_bytes() + rotator_bytes(header);
}

inline size_t doc_1bit_payload_bytes(const Header& header) {
    return one_bit_code_bytes(header) +
           one_bit_factor_bytes(header) +
           ex_factor_bytes(header);
}

inline size_t doc_4bit_payload_bytes(const Header& header) {
    return full_code_bytes(header);
}

inline size_t combined_quantized_payload_bytes(const Header& header) {
    return one_bit_code_bytes(header) +
           full_code_bytes(header) +
           one_bit_factor_bytes(header) +
           ex_factor_bytes(header);
}

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
            "Failed to write " + std::string(field_name) + " to index file: " + filename
        );
    }
}

template <typename T>
inline T read_value(std::ifstream& input, const char* field_name, const std::string& filename) {
    T value {};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error(
            "Failed to read " + std::string(field_name) + " from index file: " + filename
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
        throw std::runtime_error("Failed to write index magic to: " + filename);
    }

    write_value(output, kVersion, "version", filename);
    write_value(output, header.n, "n", filename);
    write_value(output, header.d, "d", filename);
    write_value(output, header.n_clusters, "n_clusters", filename);
    write_value(output, header.ex_bits, "ex_bits", filename);
    write_value(output, header.padded_dim, "padded_dim", filename);
    write_value(output, header.rotator_type, "rotator_type", filename);
}

inline Header read_header(std::ifstream& input, const std::string& filename) {
    char magic[sizeof(kMagic)] = {};
    input.read(magic, sizeof(magic));
    if (!input) {
        throw std::runtime_error("Failed to read index header from: " + filename);
    }
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error(
            "Legacy index format detected for " + filename +
            ". Rebuild the index so the rotator is embedded; rotator.bin is no longer used."
        );
    }

    const auto version = read_value<std::uint32_t>(input, "version", filename);
    if (version != kVersion) {
        throw std::runtime_error(
            "Unsupported index format version " + std::to_string(version) + " for " + filename
        );
    }

    Header header;
    header.n = read_value<size_t>(input, "n", filename);
    header.d = read_value<size_t>(input, "d", filename);
    header.n_clusters = read_value<size_t>(input, "n_clusters", filename);
    header.ex_bits = read_value<size_t>(input, "ex_bits", filename);
    header.padded_dim = read_value<size_t>(input, "padded_dim", filename);
    header.rotator_type =
        read_value<rabitqlib::RotatorType>(input, "rotator_type", filename);
    return header;
}

inline void validate_matching_header(
    const Header& expected,
    const Header& actual,
    const std::string& expected_filename,
    const std::string& actual_filename)
{
    if (expected.n != actual.n ||
        expected.d != actual.d ||
        expected.n_clusters != actual.n_clusters ||
        expected.ex_bits != actual.ex_bits ||
        expected.padded_dim != actual.padded_dim ||
        expected.rotator_type != actual.rotator_type) {
        throw std::runtime_error(
            "Quantized index metadata mismatch between " +
            expected_filename + " and " + actual_filename);
    }
}

inline void save_rotator(
    std::ofstream& output,
    const rabitqlib::Rotator<float>& rotator,
    const std::string& filename
) {
    rotator.save(output);
    if (!output) {
        throw std::runtime_error("Failed to write rotator to index file: " + filename);
    }
}

inline rabitqlib::Rotator<float>* load_rotator(
    std::ifstream& input,
    const Header& header,
    const std::string& filename
) {
    auto* rotator = rabitqlib::choose_rotator<float>(
        header.d,
        header.rotator_type,
        header.padded_dim
    );
    rotator->load(input);
    if (!input) {
        delete rotator;
        throw std::runtime_error("Failed to read rotator from index file: " + filename);
    }
    return rotator;
}

}  // namespace doc_format


}  // namespace Chimera
