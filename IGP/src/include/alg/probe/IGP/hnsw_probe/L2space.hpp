#pragma once
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else

#include <x86intrin.h>

#endif

#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#include <iostream>


//#include "hnswlib.hpp"

namespace hnswlib_probe
{
    using namespace std;

#ifdef USE_AVX512
    static float
        InnerProduct(const void* a, const void* b, const void* qty_ptr)
    {
        const uint32_t* dim_info = (const uint32_t*)qty_ptr;
        const uint32_t vec_dim = dim_info[0];
        const uint32_t remain_start_dimID = dim_info[1];
        constexpr uint32_t batch_n_dim = 16;

        float const* a_flt_ptr = (float const*)a;
        float const* b_flt_ptr = (float const*)b;

        __m512 sum1 = _mm512_setzero_ps();
        // n = 112;
        for (uint32_t i = 0; i + batch_n_dim < remain_start_dimID; i += batch_n_dim)
        {
            // Load two sets of 32 floats from a and b with aligned memory access
            __m512 a_vec1 = _mm512_loadu_ps(&a_flt_ptr[i]);
            __m512 b_vec1 = _mm512_loadu_ps(&b_flt_ptr[i]);
            sum1 = _mm512_fmadd_ps(a_vec1, b_vec1, sum1);
        }

        // Combine the two sum vectors to a single sum vector
        float result = 0;
        // Sum the remaining floats in the sum vector using non-vectorized operations
        for (int j = 0; j < batch_n_dim; j++)
        {
            // result += ((float*)&sum12)[j];
            result += ((float*)&sum1)[j];
        }
        for (uint32_t j = remain_start_dimID; j < vec_dim; j++)
        {
            result += a_flt_ptr[j] * b_flt_ptr[j];
        }

        return result;
    }

#elifdef USE_AVX
    inline float InnerProduct(const void* a, const void* b, const void* qty_ptr)
    {
        const uint32_t* dim_info = (const uint32_t*)qty_ptr;
        const uint32_t vec_dim = dim_info[0];
        const uint32_t remain_start_dimID = dim_info[1];
        constexpr uint32_t batch_n_dim = 8;

        float const* a_flt_ptr = static_cast<float const*>(a);
        float const* b_flt_ptr = static_cast<float const*>(b);

        __m256 sum = _mm256_setzero_ps(); // Initialize sum to 0
        for (uint32_t i = 0; i + batch_n_dim <= remain_start_dimID; i += batch_n_dim)
        {
            // Process 8 floats at a time
            __m256 a_vec = _mm256_loadu_ps(&a_flt_ptr[i]); // Load 8 floats from a
            __m256 b_vec = _mm256_loadu_ps(&b_flt_ptr[i]); // Load 8 floats from b
            __m256 sum_partial = _mm256_mul_ps(a_vec, b_vec);
            sum = _mm256_add_ps(sum, sum_partial);
        }
        float result = 0;
        for (uint32_t j = 0; j < batch_n_dim; ++j)
        {
            // Reduce sum to a single float
            result += ((float*)&sum)[j];
        }

        for (uint32_t j = remain_start_dimID; j < vec_dim; j++)
        {
            result += a_flt_ptr[j] * b_flt_ptr[j];
        }

        return result; // Return square root of sum
    }

#else
    static float
    InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr)
    {
//        printf("plain IP\n");
        uint32_t vec_dim = ((uint32_t*)qty_ptr)[0];
        float res = 0;
        for (int i = 0; i < vec_dim; i++)
        {
            float t = ((float*)pVect1)[i] * ((float*)pVect2)[i];
            res += t;
        }
        return (res);
    }
#endif

    class InnerProductSpace : public SpaceInterface<float>
    {
        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
        uint32_t dim_info[4];

    public:
        InnerProductSpace(size_t dim)
        {
            fstdistfunc_ = InnerProduct;
            data_size_ = dim * sizeof(float);
            dim_ = dim;

            uint32_t batch_n_dim;
#ifdef USE_AVX512
            batch_n_dim = 16;
#elifdef USE_AVX
            batch_n_dim = 8;
#else
            batch_n_dim = 1;
#endif

            const uint32_t n_batch = dim_ / batch_n_dim;
            const uint32_t n_remain_dim = dim_ % batch_n_dim;
            const uint32_t remain_start_dimID = batch_n_dim * n_batch;
            dim_info[0] = dim_;
            // dim_info[1] = n_batch;
            // dim_info[2] = n_remain_dim;
            // dim_info[3] = remain_start_dimID;
            dim_info[1] = remain_start_dimID;
        }

        size_t get_data_size()
        {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func()
        {
            return fstdistfunc_;
        }

        void* get_dist_func_param()
        {
            return dim_info;
        }
    };
}
