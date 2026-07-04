#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace Chimera {

struct GpuQuantizeBatchParams
{
    const float* d_rotated = nullptr;
    size_t count = 0;
    size_t dim = 0;
    uint64_t* d_one_bit_code = nullptr;
    float* d_one_bit_factor = nullptr;

    // Full codes include the sign bit plus full_ex_bits magnitude bits.
    // Supported values are 3 (4-bit payload, default) and 7 (8-bit payload).
    size_t full_ex_bits = 3;
    uint8_t* d_full_code = nullptr;
    float* d_full_factor = nullptr;

    // Optional magnitude-only sidecar. Supported values are 0 (disabled), 4, and 8.
    size_t sidecar_ex_bits = 0;
    uint8_t* d_sidecar_code = nullptr;
    float* d_sidecar_factor = nullptr;

    cudaStream_t stream = nullptr;
};

struct GpuRotateQuantizeBatchParams
{
    const float* d_raw = nullptr;
    size_t count = 0;
    size_t dim = 0;
    const uint8_t* d_fht_flip = nullptr;
    size_t fht_flip_bytes = 0;

    uint64_t* d_one_bit_code = nullptr;
    float* d_one_bit_factor = nullptr;

    size_t full_ex_bits = 3;
    uint8_t* d_full_code = nullptr;
    float* d_full_factor = nullptr;

    size_t sidecar_ex_bits = 0;
    uint8_t* d_sidecar_code = nullptr;
    float* d_sidecar_factor = nullptr;

    cudaStream_t stream = nullptr;
};

void gpu_quantize_rotated_batch(const GpuQuantizeBatchParams& params);

void gpu_rotate_quantize_batch(const GpuRotateQuantizeBatchParams& params);

}  // namespace Chimera
