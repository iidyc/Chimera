#pragma once

#include <cstddef>
#include <cstdint>
#include "query.hpp"

namespace Chimera {

float distance_one_bit(query_object* query, const char* binary_code, float one_bit_factor, size_t padded_dim);

float ip_ex_bits(query_object* query, const char* ex_code, float (*ip_func_)(const float*, const uint8_t*, size_t), size_t padded_dim);

float combine_dists(query_object* query, float one_bit_dist, float ex_bits_dist, float one_bit_factor, float ex_factor, size_t ex_bits);

float distance_ex_bits(
    query_object* query,
    const char* ex_code,
    size_t ex_bits,
    float (*ip_func_)(const float*, const uint8_t*, size_t),
    float one_bit_dist,
    float one_bit_factor,
    float ex_factor,
    size_t padded_dim
);

float distance_full_code(
    query_object* query,
    const char* full_code,
    float (*ip_func_)(const float*, const uint8_t*, size_t),
    float ex_factor,
    size_t padded_dim
);


}  // namespace Chimera
