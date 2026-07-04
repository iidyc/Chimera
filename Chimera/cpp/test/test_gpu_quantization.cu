#include "build_index/gpu_quantization.cuh"
#include "quantization.hpp"
#include "rabitqlib/utils/rotator.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kDim = 128;
constexpr size_t kCount = 64;

void cuda_check(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::vector<float> make_rotated_vectors()
{
    std::vector<float> data(kCount * kDim);
    for (size_t row = 0; row < kCount; ++row)
    {
        for (size_t dim = 0; dim < kDim; ++dim)
        {
            const int a = static_cast<int>((row * 37 + dim * 17 + 11) % 257) - 128;
            const int b = static_cast<int>((row * 13 + dim * 29 + 7) % 31) - 15;
            float value = static_cast<float>(a) * 0.013f + static_cast<float>(b) * 0.0017f;
            if (((row + dim) % 19) == 0)
            {
                value = -value;
            }
            if (value == 0.0f)
            {
                value = 0.03125f;
            }
            data[row * kDim + dim] = value;
        }
    }
    return data;
}

std::vector<float> make_raw_vectors()
{
    std::vector<float> data(kCount * kDim);
    for (size_t row = 0; row < kCount; ++row)
    {
        for (size_t dim = 0; dim < kDim; ++dim)
        {
            const int a = static_cast<int>((row * 41 + dim * 23 + 5) % 263) - 131;
            const int b = static_cast<int>((row * 19 + dim * 31 + 3) % 37) - 18;
            float value = static_cast<float>(a) * 0.011f + static_cast<float>(b) * 0.0021f;
            if (((row * 3 + dim) % 17) == 0)
            {
                value = -value;
            }
            if (value == 0.0f)
            {
                value = 0.015625f;
            }
            data[row * kDim + dim] = value;
        }
    }
    return data;
}

void require_equal_bytes(
    const std::string& label,
    const std::vector<uint8_t>& expected,
    const std::vector<uint8_t>& actual)
{
    if (expected.size() != actual.size())
    {
        throw std::runtime_error(label + " size mismatch");
    }
    for (size_t i = 0; i < expected.size(); ++i)
    {
        if (expected[i] != actual[i])
        {
            throw std::runtime_error(
                label + " byte mismatch at " + std::to_string(i) +
                ": expected=" + std::to_string(static_cast<int>(expected[i])) +
                " actual=" + std::to_string(static_cast<int>(actual[i])));
        }
    }
}

void require_equal_u64(
    const std::string& label,
    const std::vector<uint64_t>& expected,
    const std::vector<uint64_t>& actual)
{
    if (expected.size() != actual.size())
    {
        throw std::runtime_error(label + " size mismatch");
    }
    for (size_t i = 0; i < expected.size(); ++i)
    {
        if (expected[i] != actual[i])
        {
            throw std::runtime_error(
                label + " word mismatch at " + std::to_string(i));
        }
    }
}

void require_close_floats(
    const std::string& label,
    const std::vector<float>& expected,
    const std::vector<float>& actual)
{
    if (expected.size() != actual.size())
    {
        throw std::runtime_error(label + " size mismatch");
    }
    for (size_t i = 0; i < expected.size(); ++i)
    {
        constexpr float kAbsTol = 1e-6f;
        constexpr float kRelTol = 1e-5f;
        const float delta = std::fabs(expected[i] - actual[i]);
        const float scale = std::max(std::fabs(expected[i]), std::fabs(actual[i]));
        if (delta > kAbsTol && delta > kRelTol * scale)
        {
            throw std::runtime_error(
                label + " mismatch at " + std::to_string(i) +
                ": expected=" + std::to_string(expected[i]) +
                " actual=" + std::to_string(actual[i]) +
                " delta=" + std::to_string(delta));
        }
    }
}

void run_case(size_t full_ex_bits, size_t sidecar_ex_bits)
{
    const auto rotated = make_rotated_vectors();
    const size_t one_bit_words = kCount * kDim / 64;
    const size_t full_bytes = kCount * kDim * (full_ex_bits + 1) / 8;
    const size_t sidecar_bytes = kCount * kDim * sidecar_ex_bits / 8;

    std::vector<uint64_t> cpu_one_bit(one_bit_words);
    std::vector<uint8_t> cpu_full(full_bytes);
    std::vector<float> cpu_one_factor(kCount);
    std::vector<float> cpu_full_factor(kCount);
    std::vector<uint8_t> cpu_sidecar(sidecar_bytes);
    std::vector<float> cpu_sidecar_factor(sidecar_ex_bits == 0 ? 0 : kCount);

    for (size_t row = 0; row < kCount; ++row)
    {
        const float* src = rotated.data() + row * kDim;
        Chimera::encode_one_bit(
            src,
            kDim,
            cpu_one_bit.data() + row * (kDim / 64),
            &cpu_one_factor[row]);
        Chimera::encode_full_code(
            src,
            kDim,
            full_ex_bits,
            cpu_full.data() + row * (kDim * (full_ex_bits + 1) / 8),
            &cpu_full_factor[row]);
        if (sidecar_ex_bits != 0)
        {
            Chimera::encode_ex_bits(
                src,
                kDim,
                sidecar_ex_bits,
                cpu_sidecar.data() + row * (kDim * sidecar_ex_bits / 8),
                &cpu_sidecar_factor[row]);
        }
    }

    float* d_rotated = nullptr;
    uint64_t* d_one_bit = nullptr;
    uint8_t* d_full = nullptr;
    float* d_one_factor = nullptr;
    float* d_full_factor = nullptr;
    uint8_t* d_sidecar = nullptr;
    float* d_sidecar_factor = nullptr;

    cuda_check(cudaMalloc(&d_rotated, rotated.size() * sizeof(float)), "cudaMalloc rotated");
    cuda_check(cudaMalloc(&d_one_bit, cpu_one_bit.size() * sizeof(uint64_t)), "cudaMalloc one_bit");
    cuda_check(cudaMalloc(&d_full, cpu_full.size()), "cudaMalloc full");
    cuda_check(cudaMalloc(&d_one_factor, cpu_one_factor.size() * sizeof(float)), "cudaMalloc one_factor");
    cuda_check(cudaMalloc(&d_full_factor, cpu_full_factor.size() * sizeof(float)), "cudaMalloc full_factor");
    if (sidecar_ex_bits != 0)
    {
        cuda_check(cudaMalloc(&d_sidecar, cpu_sidecar.size()), "cudaMalloc sidecar");
        cuda_check(cudaMalloc(&d_sidecar_factor, cpu_sidecar_factor.size() * sizeof(float)), "cudaMalloc sidecar_factor");
    }

    try
    {
        cuda_check(
            cudaMemcpy(d_rotated, rotated.data(), rotated.size() * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy rotated");

        Chimera::GpuQuantizeBatchParams params;
        params.d_rotated = d_rotated;
        params.count = kCount;
        params.dim = kDim;
        params.d_one_bit_code = d_one_bit;
        params.d_one_bit_factor = d_one_factor;
        params.full_ex_bits = full_ex_bits;
        params.d_full_code = d_full;
        params.d_full_factor = d_full_factor;
        params.sidecar_ex_bits = sidecar_ex_bits;
        params.d_sidecar_code = d_sidecar;
        params.d_sidecar_factor = d_sidecar_factor;
        Chimera::gpu_quantize_rotated_batch(params);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        std::vector<uint64_t> gpu_one_bit(cpu_one_bit.size());
        std::vector<uint8_t> gpu_full(cpu_full.size());
        std::vector<float> gpu_one_factor(cpu_one_factor.size());
        std::vector<float> gpu_full_factor(cpu_full_factor.size());
        std::vector<uint8_t> gpu_sidecar(cpu_sidecar.size());
        std::vector<float> gpu_sidecar_factor(cpu_sidecar_factor.size());

        cuda_check(cudaMemcpy(gpu_one_bit.data(), d_one_bit, gpu_one_bit.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy one_bit");
        cuda_check(cudaMemcpy(gpu_full.data(), d_full, gpu_full.size(), cudaMemcpyDeviceToHost), "copy full");
        cuda_check(cudaMemcpy(gpu_one_factor.data(), d_one_factor, gpu_one_factor.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy one_factor");
        cuda_check(cudaMemcpy(gpu_full_factor.data(), d_full_factor, gpu_full_factor.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy full_factor");
        if (sidecar_ex_bits != 0)
        {
            cuda_check(cudaMemcpy(gpu_sidecar.data(), d_sidecar, gpu_sidecar.size(), cudaMemcpyDeviceToHost), "copy sidecar");
            cuda_check(cudaMemcpy(gpu_sidecar_factor.data(), d_sidecar_factor, gpu_sidecar_factor.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy sidecar_factor");
        }

        const std::string suffix =
            " full_ex_bits=" + std::to_string(full_ex_bits) +
            " sidecar_ex_bits=" + std::to_string(sidecar_ex_bits);
        require_equal_u64("one_bit" + suffix, cpu_one_bit, gpu_one_bit);
        require_equal_bytes("full_code" + suffix, cpu_full, gpu_full);
        require_close_floats("one_factor" + suffix, cpu_one_factor, gpu_one_factor);
        require_close_floats("full_factor" + suffix, cpu_full_factor, gpu_full_factor);
        if (sidecar_ex_bits != 0)
        {
            require_equal_bytes("sidecar_code" + suffix, cpu_sidecar, gpu_sidecar);
            require_close_floats("sidecar_factor" + suffix, cpu_sidecar_factor, gpu_sidecar_factor);
        }
    }
    catch (...)
    {
        cudaFree(d_rotated);
        cudaFree(d_one_bit);
        cudaFree(d_full);
        cudaFree(d_one_factor);
        cudaFree(d_full_factor);
        cudaFree(d_sidecar);
        cudaFree(d_sidecar_factor);
        throw;
    }

    cudaFree(d_rotated);
    cudaFree(d_one_bit);
    cudaFree(d_full);
    cudaFree(d_one_factor);
    cudaFree(d_full_factor);
    cudaFree(d_sidecar);
    cudaFree(d_sidecar_factor);
}

void run_raw_case(size_t full_ex_bits, size_t sidecar_ex_bits)
{
    const auto raw = make_raw_vectors();
    auto* rotator = rabitqlib::choose_rotator_with_seed<float>(
        kDim,
        rabitqlib::RotatorType::FhtKacRotator,
        kDim,
        42);
    const auto* fht_rotator =
        dynamic_cast<const rabitqlib::rotator_impl::FhtKacRotator*>(rotator);
    if (fht_rotator == nullptr)
    {
        delete rotator;
        throw std::runtime_error("expected FhtKacRotator");
    }

    std::vector<float> rotated(kCount * kDim);
    for (size_t row = 0; row < kCount; ++row)
    {
        rotator->rotate(raw.data() + row * kDim, rotated.data() + row * kDim);
    }

    const size_t one_bit_words = kCount * kDim / 64;
    const size_t full_bytes = kCount * kDim * (full_ex_bits + 1) / 8;
    const size_t sidecar_bytes = kCount * kDim * sidecar_ex_bits / 8;

    std::vector<uint64_t> cpu_one_bit(one_bit_words);
    std::vector<uint8_t> cpu_full(full_bytes);
    std::vector<float> cpu_one_factor(kCount);
    std::vector<float> cpu_full_factor(kCount);
    std::vector<uint8_t> cpu_sidecar(sidecar_bytes);
    std::vector<float> cpu_sidecar_factor(sidecar_ex_bits == 0 ? 0 : kCount);

    for (size_t row = 0; row < kCount; ++row)
    {
        const float* src = rotated.data() + row * kDim;
        Chimera::encode_one_bit(
            src,
            kDim,
            cpu_one_bit.data() + row * (kDim / 64),
            &cpu_one_factor[row]);
        Chimera::encode_full_code(
            src,
            kDim,
            full_ex_bits,
            cpu_full.data() + row * (kDim * (full_ex_bits + 1) / 8),
            &cpu_full_factor[row]);
        if (sidecar_ex_bits != 0)
        {
            Chimera::encode_ex_bits(
                src,
                kDim,
                sidecar_ex_bits,
                cpu_sidecar.data() + row * (kDim * sidecar_ex_bits / 8),
                &cpu_sidecar_factor[row]);
        }
    }

    float* d_raw = nullptr;
    uint8_t* d_flip = nullptr;
    uint64_t* d_one_bit = nullptr;
    uint8_t* d_full = nullptr;
    float* d_one_factor = nullptr;
    float* d_full_factor = nullptr;
    uint8_t* d_sidecar = nullptr;
    float* d_sidecar_factor = nullptr;

    cuda_check(cudaMalloc(&d_raw, raw.size() * sizeof(float)), "cudaMalloc raw");
    cuda_check(cudaMalloc(&d_flip, fht_rotator->flip_bytes()), "cudaMalloc flip");
    cuda_check(cudaMalloc(&d_one_bit, cpu_one_bit.size() * sizeof(uint64_t)), "cudaMalloc one_bit");
    cuda_check(cudaMalloc(&d_full, cpu_full.size()), "cudaMalloc full");
    cuda_check(cudaMalloc(&d_one_factor, cpu_one_factor.size() * sizeof(float)), "cudaMalloc one_factor");
    cuda_check(cudaMalloc(&d_full_factor, cpu_full_factor.size() * sizeof(float)), "cudaMalloc full_factor");
    if (sidecar_ex_bits != 0)
    {
        cuda_check(cudaMalloc(&d_sidecar, cpu_sidecar.size()), "cudaMalloc sidecar");
        cuda_check(cudaMalloc(&d_sidecar_factor, cpu_sidecar_factor.size() * sizeof(float)), "cudaMalloc sidecar_factor");
    }

    try
    {
        cuda_check(cudaMemcpy(d_raw, raw.data(), raw.size() * sizeof(float), cudaMemcpyHostToDevice), "copy raw");
        cuda_check(cudaMemcpy(d_flip, fht_rotator->flip_data(), fht_rotator->flip_bytes(), cudaMemcpyHostToDevice), "copy flip");

        Chimera::GpuRotateQuantizeBatchParams params;
        params.d_raw = d_raw;
        params.count = kCount;
        params.dim = kDim;
        params.d_fht_flip = d_flip;
        params.fht_flip_bytes = fht_rotator->flip_bytes();
        params.d_one_bit_code = d_one_bit;
        params.d_one_bit_factor = d_one_factor;
        params.full_ex_bits = full_ex_bits;
        params.d_full_code = d_full;
        params.d_full_factor = d_full_factor;
        params.sidecar_ex_bits = sidecar_ex_bits;
        params.d_sidecar_code = d_sidecar;
        params.d_sidecar_factor = d_sidecar_factor;
        Chimera::gpu_rotate_quantize_batch(params);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize raw");

        std::vector<uint64_t> gpu_one_bit(cpu_one_bit.size());
        std::vector<uint8_t> gpu_full(cpu_full.size());
        std::vector<float> gpu_one_factor(cpu_one_factor.size());
        std::vector<float> gpu_full_factor(cpu_full_factor.size());
        std::vector<uint8_t> gpu_sidecar(cpu_sidecar.size());
        std::vector<float> gpu_sidecar_factor(cpu_sidecar_factor.size());

        cuda_check(cudaMemcpy(gpu_one_bit.data(), d_one_bit, gpu_one_bit.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy one_bit");
        cuda_check(cudaMemcpy(gpu_full.data(), d_full, gpu_full.size(), cudaMemcpyDeviceToHost), "copy full");
        cuda_check(cudaMemcpy(gpu_one_factor.data(), d_one_factor, gpu_one_factor.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy one_factor");
        cuda_check(cudaMemcpy(gpu_full_factor.data(), d_full_factor, gpu_full_factor.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy full_factor");
        if (sidecar_ex_bits != 0)
        {
            cuda_check(cudaMemcpy(gpu_sidecar.data(), d_sidecar, gpu_sidecar.size(), cudaMemcpyDeviceToHost), "copy sidecar");
            cuda_check(cudaMemcpy(gpu_sidecar_factor.data(), d_sidecar_factor, gpu_sidecar_factor.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy sidecar_factor");
        }

        const std::string suffix =
            " raw full_ex_bits=" + std::to_string(full_ex_bits) +
            " sidecar_ex_bits=" + std::to_string(sidecar_ex_bits);
        require_equal_u64("one_bit" + suffix, cpu_one_bit, gpu_one_bit);
        require_close_floats("one_factor" + suffix, cpu_one_factor, gpu_one_factor);
        require_close_floats("full_factor" + suffix, cpu_full_factor, gpu_full_factor);
        if (sidecar_ex_bits != 0)
        {
            require_close_floats("sidecar_factor" + suffix, cpu_sidecar_factor, gpu_sidecar_factor);
        }
    }
    catch (...)
    {
        cudaFree(d_raw);
        cudaFree(d_flip);
        cudaFree(d_one_bit);
        cudaFree(d_full);
        cudaFree(d_one_factor);
        cudaFree(d_full_factor);
        cudaFree(d_sidecar);
        cudaFree(d_sidecar_factor);
        delete rotator;
        throw;
    }

    cudaFree(d_raw);
    cudaFree(d_flip);
    cudaFree(d_one_bit);
    cudaFree(d_full);
    cudaFree(d_one_factor);
    cudaFree(d_full_factor);
    cudaFree(d_sidecar);
    cudaFree(d_sidecar_factor);
    delete rotator;
}

}  // namespace

int main()
{
    try
    {
        run_case(3, 0);
        run_case(3, 4);
        run_case(7, 0);
        run_case(7, 8);
        run_raw_case(3, 0);
        run_raw_case(3, 4);
        run_raw_case(7, 0);
        run_raw_case(7, 8);
        std::cout << "GPU quantization CPU comparison passed." << std::endl;
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "GPU quantization test failed: " << e.what() << std::endl;
        return 1;
    }
}
