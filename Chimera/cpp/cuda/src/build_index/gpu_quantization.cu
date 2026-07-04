#include "build_index/gpu_quantization.cuh"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace Chimera {
namespace {

constexpr size_t kSupportedDim = 128;
constexpr double kEps = 1e-5;
constexpr int kNEnum = 10;

__device__ __constant__ float kTightStartDevice[9] = {
    0.0f,
    0.15f,
    0.20f,
    0.52f,
    0.59f,
    0.71f,
    0.75f,
    0.77f,
    0.81f,
};

void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ double best_rescale_factor_device(const float* o_abs, size_t dim, size_t ex_bits)
{
    double max_o = static_cast<double>(o_abs[0]);
    for (size_t i = 1; i < dim; ++i)
    {
        max_o = fmax(max_o, static_cast<double>(o_abs[i]));
    }

    const double t_end =
        static_cast<double>(((1 << ex_bits) - 1) + kNEnum) / max_o;
    const double t_start =
        t_end * static_cast<double>(kTightStartDevice[ex_bits]);

    int cur_o_bar[kSupportedDim];
    double next_t[kSupportedDim];
    bool active[kSupportedDim];
    double sqr_denominator = static_cast<double>(dim) * 0.25;
    double numerator = 0.0;

    for (size_t i = 0; i < dim; ++i)
    {
        const int cur = static_cast<int>((t_start * static_cast<double>(o_abs[i])) + kEps);
        cur_o_bar[i] = cur;
        sqr_denominator += static_cast<double>(cur * cur + cur);
        numerator += (static_cast<double>(cur) + 0.5) * static_cast<double>(o_abs[i]);
        next_t[i] = static_cast<double>(cur + 1) / static_cast<double>(o_abs[i]);
        active[i] = true;
    }

    double max_ip = 0.0;
    double t = 0.0;

    while (true)
    {
        bool found = false;
        double cur_t = 0.0;
        size_t update_id = 0;
        for (size_t i = 0; i < dim; ++i)
        {
            if (!active[i])
            {
                continue;
            }
            if (!found || next_t[i] < cur_t || (next_t[i] == cur_t && i < update_id))
            {
                found = true;
                cur_t = next_t[i];
                update_id = i;
            }
        }
        if (!found)
        {
            break;
        }

        active[update_id] = false;
        cur_o_bar[update_id]++;
        const int update_o_bar = cur_o_bar[update_id];
        sqr_denominator += static_cast<double>(2 * update_o_bar);
        numerator += static_cast<double>(o_abs[update_id]);

        const double cur_ip = numerator / sqrt(sqr_denominator);
        if (cur_ip > max_ip)
        {
            max_ip = cur_ip;
            t = cur_t;
        }

        if (update_o_bar < (1 << ex_bits) - 1)
        {
            const double t_next =
                static_cast<double>(update_o_bar + 1) / static_cast<double>(o_abs[update_id]);
            if (t_next < t_end)
            {
                next_t[update_id] = t_next;
                active[update_id] = true;
            }
        }
    }

    return t;
}

__device__ float eigen_reduce_128_device(const float* terms);

__device__ void quantize_ex_device(
    const float* rotated_data,
    uint8_t* ex_code,
    size_t dim,
    size_t ex_bits)
{
    float norm_terms[kSupportedDim];
    for (size_t i = 0; i < dim; ++i)
    {
        norm_terms[i] = __fmul_rn(rotated_data[i], rotated_data[i]);
    }
    const float norm_sqr = eigen_reduce_128_device(norm_terms);
    const float inv_norm = 1.0f / sqrtf(norm_sqr);

    float abs_data[kSupportedDim];
    for (size_t i = 0; i < dim; ++i)
    {
        abs_data[i] = fabsf(rotated_data[i] * inv_norm);
    }

    const double t = best_rescale_factor_device(abs_data, dim, ex_bits);
    const int mask = (1 << ex_bits) - 1;
    for (size_t i = 0; i < dim; ++i)
    {
        int code = static_cast<int>((t * static_cast<double>(abs_data[i])) + kEps);
        if (code >= (1 << ex_bits))
        {
            code = (1 << ex_bits) - 1;
        }
        uint8_t stored = static_cast<uint8_t>(code);
        if (rotated_data[i] < 0.0f)
        {
            stored = static_cast<uint8_t>((~stored) & mask);
        }
        ex_code[i] = stored;
    }
}

__device__ void pack_4bit_code_device(const uint8_t* raw_code, uint8_t* compact_code, size_t dim)
{
    for (size_t j = 0; j < dim; j += 16)
    {
        uint64_t code0 = 0;
        uint64_t code1 = 0;
        for (size_t i = 0; i < 8; ++i)
        {
            code0 |= static_cast<uint64_t>(raw_code[j + i]) << (8 * i);
            code1 |= static_cast<uint64_t>(raw_code[j + 8 + i]) << (8 * i);
        }
        const uint64_t compact = (code1 << 4) | code0;
        for (size_t b = 0; b < 8; ++b)
        {
            compact_code[j / 2 + b] = static_cast<uint8_t>((compact >> (8 * b)) & 0xFFu);
        }
    }
}

__device__ void pack_8bit_code_device(const uint8_t* raw_code, uint8_t* compact_code, size_t dim)
{
    for (size_t i = 0; i < dim; ++i)
    {
        compact_code[i] = raw_code[i];
    }
}

__device__ float eigen_reduce_128_device(const float* terms)
{
    float packet_res0[16];
    float packet_res1[16];
    for (size_t lane = 0; lane < 16; ++lane)
    {
        packet_res0[lane] = terms[lane];
        packet_res1[lane] = terms[16 + lane];
    }

    for (size_t base = 32; base < 128; base += 32)
    {
        for (size_t lane = 0; lane < 16; ++lane)
        {
            packet_res0[lane] = __fadd_rn(packet_res0[lane], terms[base + lane]);
            packet_res1[lane] = __fadd_rn(packet_res1[lane], terms[base + 16 + lane]);
        }
    }

    float packet[16];
    for (size_t lane = 0; lane < 16; ++lane)
    {
        packet[lane] = __fadd_rn(packet_res0[lane], packet_res1[lane]);
    }

    float lane8[8];
    for (size_t lane = 0; lane < 8; ++lane)
    {
        lane8[lane] = __fadd_rn(packet[lane], packet[lane + 8]);
    }

    float lane4[4];
    for (size_t lane = 0; lane < 4; ++lane)
    {
        lane4[lane] = __fadd_rn(lane8[lane], lane8[lane + 4]);
    }

    const float tmp0 = __fadd_rn(lane4[0], lane4[2]);
    const float tmp1 = __fadd_rn(lane4[1], lane4[3]);
    return __fadd_rn(tmp0, tmp1);
}

__device__ void encode_ex_payload_device(
    const float* x,
    size_t ex_bits,
    uint8_t* compact_code,
    float* factor,
    bool pack_sign_bit)
{
    uint8_t ex_code[kSupportedDim];
    quantize_ex_device(x, ex_code, kSupportedDim, ex_bits);

    uint8_t packed_total_code[kSupportedDim];
    float terms[kSupportedDim];
    const float cb = -(static_cast<float>(1 << ex_bits) - 0.5f);
    for (size_t i = 0; i < kSupportedDim; ++i)
    {
        const uint8_t sign = x[i] >= 0.0f ? 1 : 0;
        const int total = static_cast<int>(ex_code[i]) +
            (static_cast<int>(sign) << ex_bits);
        packed_total_code[i] = static_cast<uint8_t>(total);
        terms[i] = __fmul_rn(static_cast<float>(total) + cb, x[i]);
    }
    const float ip = eigen_reduce_128_device(terms);
    *factor = 1.0f / ip;

    const uint8_t* packed_code = pack_sign_bit ? packed_total_code : ex_code;
    const size_t packed_bits = pack_sign_bit ? ex_bits + 1 : ex_bits;
    if (packed_bits == 4)
    {
        pack_4bit_code_device(packed_code, compact_code, kSupportedDim);
    }
    else if (packed_bits == 8)
    {
        pack_8bit_code_device(packed_code, compact_code, kSupportedDim);
    }
}

__device__ float flip_sign_device(float value, const uint8_t* flip, size_t dim)
{
    const uint8_t mask = static_cast<uint8_t>(1u << (dim & 7u));
    if ((flip[dim >> 3u] & mask) == 0)
    {
        return value;
    }
    return __uint_as_float(__float_as_uint(value) ^ 0x80000000u);
}

__device__ void fht128_device(float* values)
{
    for (size_t half = 1; half < kSupportedDim; half <<= 1)
    {
        const size_t step = half << 1;
        for (size_t base = 0; base < kSupportedDim; base += step)
        {
            for (size_t i = 0; i < half; ++i)
            {
                const float a = values[base + i];
                const float b = values[base + i + half];
                values[base + i] = __fadd_rn(a, b);
                values[base + i + half] = __fsub_rn(a, b);
            }
        }
    }
}

__device__ void rotate_fht128_device(
    const float* raw,
    const uint8_t* flip,
    float* rotated)
{
    constexpr float kScale = 0.08838834764831845f;  // 1 / sqrt(128)
    constexpr size_t kFlipBytesPerRound = kSupportedDim / 8;

    for (size_t i = 0; i < kSupportedDim; ++i)
    {
        rotated[i] = raw[i];
    }

    for (size_t round = 0; round < 4; ++round)
    {
        const uint8_t* round_flip = flip + round * kFlipBytesPerRound;
        for (size_t i = 0; i < kSupportedDim; ++i)
        {
            rotated[i] = flip_sign_device(rotated[i], round_flip, i);
        }
        fht128_device(rotated);
        for (size_t i = 0; i < kSupportedDim; ++i)
        {
            rotated[i] = __fmul_rn(rotated[i], kScale);
        }
    }
}

__global__ void quantize_rotated_kernel(
    const float* __restrict__ rotated,
    size_t count,
    size_t full_ex_bits,
    size_t sidecar_ex_bits,
    uint64_t* __restrict__ one_bit_code,
    uint8_t* __restrict__ full_code,
    float* __restrict__ one_bit_factor,
    float* __restrict__ full_factor,
    uint8_t* __restrict__ sidecar_code,
    float* __restrict__ sidecar_factor)
{
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count)
    {
        return;
    }

    const float* x = rotated + row * kSupportedDim;
    uint64_t* one_bit_out = one_bit_code + row * (kSupportedDim / 64);
    const size_t full_bytes_per_vector = kSupportedDim * (full_ex_bits + 1) / 8;
    uint8_t* full_code_out = full_code + row * full_bytes_per_vector;

    float one_bit_terms[kSupportedDim];
    for (size_t word = 0; word < kSupportedDim / 64; ++word)
    {
        uint64_t packed = 0;
        for (size_t bit = 0; bit < 64; ++bit)
        {
            const size_t dim = word * 64 + bit;
            const int sign = x[dim] > 0.0f ? 1 : 0;
            packed |= static_cast<uint64_t>(sign) << bit;
            one_bit_terms[dim] = __fmul_rn(static_cast<float>(sign) - 0.5f, x[dim]);
        }
        one_bit_out[word] = packed;
    }
    const float one_bit_ip = eigen_reduce_128_device(one_bit_terms);
    one_bit_factor[row] = 1.0f / one_bit_ip;

    encode_ex_payload_device(
        x,
        full_ex_bits,
        full_code_out,
        &full_factor[row],
        true);

    if (sidecar_ex_bits != 0)
    {
        const size_t sidecar_bytes_per_vector = kSupportedDim * sidecar_ex_bits / 8;
        encode_ex_payload_device(
            x,
            sidecar_ex_bits,
            sidecar_code + row * sidecar_bytes_per_vector,
            &sidecar_factor[row],
            false);
    }
}

__global__ void rotate_quantize_kernel(
    const float* __restrict__ raw,
    const uint8_t* __restrict__ flip,
    size_t count,
    size_t full_ex_bits,
    size_t sidecar_ex_bits,
    uint64_t* __restrict__ one_bit_code,
    uint8_t* __restrict__ full_code,
    float* __restrict__ one_bit_factor,
    float* __restrict__ full_factor,
    uint8_t* __restrict__ sidecar_code,
    float* __restrict__ sidecar_factor)
{
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count)
    {
        return;
    }

    float rotated[kSupportedDim];
    rotate_fht128_device(raw + row * kSupportedDim, flip, rotated);

    uint64_t* one_bit_out = one_bit_code + row * (kSupportedDim / 64);
    const size_t full_bytes_per_vector = kSupportedDim * (full_ex_bits + 1) / 8;
    uint8_t* full_code_out = full_code + row * full_bytes_per_vector;

    float one_bit_terms[kSupportedDim];
    for (size_t word = 0; word < kSupportedDim / 64; ++word)
    {
        uint64_t packed = 0;
        for (size_t bit = 0; bit < 64; ++bit)
        {
            const size_t dim = word * 64 + bit;
            const int sign = rotated[dim] > 0.0f ? 1 : 0;
            packed |= static_cast<uint64_t>(sign) << bit;
            one_bit_terms[dim] =
                __fmul_rn(static_cast<float>(sign) - 0.5f, rotated[dim]);
        }
        one_bit_out[word] = packed;
    }
    const float one_bit_ip = eigen_reduce_128_device(one_bit_terms);
    one_bit_factor[row] = 1.0f / one_bit_ip;

    encode_ex_payload_device(
        rotated,
        full_ex_bits,
        full_code_out,
        &full_factor[row],
        true);

    if (sidecar_ex_bits != 0)
    {
        const size_t sidecar_bytes_per_vector = kSupportedDim * sidecar_ex_bits / 8;
        encode_ex_payload_device(
            rotated,
            sidecar_ex_bits,
            sidecar_code + row * sidecar_bytes_per_vector,
            &sidecar_factor[row],
            false);
    }
}

}  // namespace

void gpu_quantize_rotated_batch(const GpuQuantizeBatchParams& params)
{
    if (params.count == 0)
    {
        return;
    }
    if (params.d_rotated == nullptr || params.d_one_bit_code == nullptr ||
        params.d_full_code == nullptr || params.d_one_bit_factor == nullptr ||
        params.d_full_factor == nullptr)
    {
        throw std::runtime_error("gpu_quantize_rotated_batch received null buffer");
    }
    if (params.dim != kSupportedDim)
    {
        throw std::runtime_error("gpu_quantize_rotated_batch currently requires dim=128");
    }
    if (params.full_ex_bits != 3 && params.full_ex_bits != 7)
    {
        throw std::runtime_error(
            "gpu_quantize_rotated_batch supports full_ex_bits=3 or 7");
    }
    if (params.sidecar_ex_bits != 0 &&
        params.sidecar_ex_bits != 4 &&
        params.sidecar_ex_bits != 8)
    {
        throw std::runtime_error(
            "gpu_quantize_rotated_batch supports sidecar_ex_bits=0, 4, or 8");
    }
    if (params.sidecar_ex_bits != 0 &&
        (params.d_sidecar_code == nullptr || params.d_sidecar_factor == nullptr))
    {
        throw std::runtime_error(
            "gpu_quantize_rotated_batch sidecar buffers are required when sidecar_ex_bits is set");
    }

    constexpr int kThreads = 128;
    const int blocks = static_cast<int>((params.count + kThreads - 1) / kThreads);
    quantize_rotated_kernel<<<blocks, kThreads, 0, params.stream>>>(
        params.d_rotated,
        params.count,
        params.full_ex_bits,
        params.sidecar_ex_bits,
        params.d_one_bit_code,
        params.d_full_code,
        params.d_one_bit_factor,
        params.d_full_factor,
        params.d_sidecar_code,
        params.d_sidecar_factor);
    check_cuda(cudaGetLastError(), "gpu_quantize_rotated_batch launch failed");
}

void gpu_rotate_quantize_batch(const GpuRotateQuantizeBatchParams& params)
{
    if (params.count == 0)
    {
        return;
    }
    if (params.d_raw == nullptr || params.d_fht_flip == nullptr ||
        params.d_one_bit_code == nullptr || params.d_full_code == nullptr ||
        params.d_one_bit_factor == nullptr || params.d_full_factor == nullptr)
    {
        throw std::runtime_error("gpu_rotate_quantize_batch received null buffer");
    }
    if (params.dim != kSupportedDim)
    {
        throw std::runtime_error("gpu_rotate_quantize_batch currently requires dim=128");
    }
    if (params.fht_flip_bytes != 4 * kSupportedDim / 8)
    {
        throw std::runtime_error("gpu_rotate_quantize_batch received invalid FHT flip size");
    }
    if (params.full_ex_bits != 3 && params.full_ex_bits != 7)
    {
        throw std::runtime_error(
            "gpu_rotate_quantize_batch supports full_ex_bits=3 or 7");
    }
    if (params.sidecar_ex_bits != 0 &&
        params.sidecar_ex_bits != 4 &&
        params.sidecar_ex_bits != 8)
    {
        throw std::runtime_error(
            "gpu_rotate_quantize_batch supports sidecar_ex_bits=0, 4, or 8");
    }
    if (params.sidecar_ex_bits != 0 &&
        (params.d_sidecar_code == nullptr || params.d_sidecar_factor == nullptr))
    {
        throw std::runtime_error(
            "gpu_rotate_quantize_batch sidecar buffers are required when sidecar_ex_bits is set");
    }

    constexpr int kThreads = 128;
    const int blocks = static_cast<int>((params.count + kThreads - 1) / kThreads);
    rotate_quantize_kernel<<<blocks, kThreads, 0, params.stream>>>(
        params.d_raw,
        params.d_fht_flip,
        params.count,
        params.full_ex_bits,
        params.sidecar_ex_bits,
        params.d_one_bit_code,
        params.d_full_code,
        params.d_one_bit_factor,
        params.d_full_factor,
        params.d_sidecar_code,
        params.d_sidecar_factor);
    check_cuda(cudaGetLastError(), "gpu_rotate_quantize_batch launch failed");
}

}  // namespace Chimera
