#pragma once

#include <numeric>
#include <cstddef>

namespace Chimera {

struct query_object {
    float* rotated_query;
    float cb1_sumq;
    float cbex_sumq;

    query_object() = default;

    explicit query_object(float* rotated_query, size_t padded_dim, size_t ex_bits);
};


}  // namespace Chimera
