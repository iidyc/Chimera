#pragma once

#include <cstddef>
#include <cstdint>

void pack_binary(const int* binary_code, uint64_t* compact_code, size_t length);

void encode_one_bit(const float* rotated_data, size_t dim, uint64_t* one_bit_code, float* factor);

double best_rescale_factor(const float* o_abs, size_t dim, size_t ex_bits);

void quantize_ex(const float* rotated_data, uint8_t* ex_code, size_t dim, size_t ex_bits);

void encode_ex_bits(const float* rotated_data, size_t dim, size_t ex_bits, uint8_t* compact_ex_code, float* factor);
