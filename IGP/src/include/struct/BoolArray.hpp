//
// Created by 13172 on 2024/5/28.
//

#ifndef VECTORSETSEARCH_BOOLARRAY_HPP
#define VECTORSETSEARCH_BOOLARRAY_HPP

#include <vector>
#include <unordered_map>
#include <memory>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace VectorSetSearch {

    class BoolArray {
        static constexpr uint64_t n_bit_per_byte_ = 8;
        size_t n_char_;
        std::unique_ptr<unsigned char[]> array_;

    public:

        BoolArray() = default;

        BoolArray(size_t n_val, bool init_val)
                : n_char_((n_val + n_bit_per_byte_ - 1) / n_bit_per_byte_),
                  array_(std::make_unique<unsigned char[]>(n_char_)) {
            reset(init_val);
        }

        void reset(bool init_val) {
            if (init_val) {
                std::memset(array_.get(), 0xFF, n_char_);
            } else {
                std::memset(array_.get(), 0x00, n_char_);
            }
        }

        bool operator[](size_t idx) const {
            auto char_idx = idx / n_bit_per_byte_;
            auto offset = idx % n_bit_per_byte_;
            return array_[char_idx] & (1 << offset);
        }

        void set(size_t idx, bool value) {
            auto char_idx = idx / n_bit_per_byte_;
            auto offset = idx % n_bit_per_byte_;
            assert(0 <= offset && offset < 8);

            if (value) {
                array_[char_idx] |= 1 << offset;
            } else {
                array_[char_idx] &= ~(1 << offset);
            }
        }
    };

}

#endif //VECTORSETSEARCH_BOOLARRAY_HPP
