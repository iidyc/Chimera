#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>
#include <vector>

namespace Chimera {

template <typename T, std::size_t Alignment>
class cpu_aligned_allocator {
   public:
    using value_type = T;

    cpu_aligned_allocator() noexcept = default;

    template <typename U>
    cpu_aligned_allocator(const cpu_aligned_allocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n)
    {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept
    {
        std::free(p);
    }

    template <typename U>
    struct rebind {
        using other = cpu_aligned_allocator<U, Alignment>;
    };
};

template <typename T1, typename T2, std::size_t Alignment>
bool operator==(const cpu_aligned_allocator<T1, Alignment>&,
                const cpu_aligned_allocator<T2, Alignment>&) noexcept
{
    return true;
}

template <typename T1, typename T2, std::size_t Alignment>
bool operator!=(const cpu_aligned_allocator<T1, Alignment>&,
                const cpu_aligned_allocator<T2, Alignment>&) noexcept
{
    return false;
}

template <typename T>
using aligned_vector_64 = std::vector<T, cpu_aligned_allocator<T, 64>>;


}  // namespace Chimera
