#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "io.hpp"

// ---------------------------------------------------------------------------
// GPU-accelerated index build pipeline
// ---------------------------------------------------------------------------
// 1. Randomly sample centroids
// 2. Build a temporary non-rotated CAGRA graph for fast centroid assignment
// 3. Build the persisted CAGRA graph on rotated centroids
// 4. Assemble IVF_PG, quantize data, and serialize the selected output set into
//    `index_dir`. The build mode controls HNSW/metadata artifacts; the
//    quantization mode controls full-code bit width and optional sidecar output.
//
// ---------------------------------------------------------------------------

namespace Chimera {

struct CentroidSampleOptions
{
    size_t bucket_size = 256;
    std::uint64_t seed = 41;
};

enum class BuildOutputMode
{
    Full,
    GpuSearchMinimal
};

enum class QuantizationMode
{
    Bit4,
    Bit4Ex,
    Bit8,
    Bit8Ex
};

enum class QuantizationBackend
{
    Cpu,
    Gpu,
    GpuMerged
};

struct BuildIndexOptions
{
    CentroidSampleOptions centroid_sample;
    BuildOutputMode output_mode = BuildOutputMode::Full;
    QuantizationMode quantization_mode = QuantizationMode::Bit4;
    QuantizationBackend quantization_backend = QuantizationBackend::Cpu;
};

BuildOutputMode parse_build_output_mode(const std::string& value);

const char* build_output_mode_name(BuildOutputMode mode);

QuantizationMode parse_quantization_mode(const std::string& value);

const char* quantization_mode_name(QuantizationMode mode);

size_t quantization_full_ex_bits(QuantizationMode mode);

size_t quantization_sidecar_ex_bits(QuantizationMode mode);

QuantizationBackend parse_quantization_backend(const std::string& value);

const char* quantization_backend_name(QuantizationBackend backend);

void build_index(
    const MmapedEmbeddings& data,
    size_t n_clusters,
    size_t ex_bits,
    const std::vector<int>& doc_lens,
    const std::string& index_dir,
    const CentroidSampleOptions& centroid_sample_options = {});

void build_index(
    const MmapedEmbeddings& data,
    size_t n_clusters,
    size_t ex_bits,
    const std::vector<int>& doc_lens,
    const std::string& index_dir,
    const BuildIndexOptions& options);

}  // namespace Chimera
