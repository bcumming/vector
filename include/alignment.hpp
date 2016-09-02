#pragma once

#include <stdexcept>
#include <cstdint>

namespace memory {
namespace util {

/// returns
///     true if ptr is aligned on n byte boundary
///     false otherwise
/// notes
///     - use convert to void* because the standard only guarentees
///       conversion to uintptr_t for void*
template <typename T>
bool is_aligned(const T* ptr, std::size_t n) {
    //std::cout << "testing " << std::uintptr_t( (const void*)(ptr) ) << " for alignment " << n << " : " << std::uintptr_t( (const void*)(ptr) )%n << std::endl;
    return std::uintptr_t( (const void*)(ptr) )%n == 0;
}

/// check if offset from an aligned address is also aligned
/// the offset is in items of type T
template <typename T>
constexpr bool is_aligned(std::size_t offset, std::size_t alignment) {
    return (offset*sizeof(T))%alignment == 0;
}

class alignment_error: public std::runtime_error {

public:
    template <typename S>
    alignment_error(S&& whatmsg) :
        std::runtime_error(std::forward<S>(whatmsg))
    {}
};

} // namespace util
} // namespace memory
