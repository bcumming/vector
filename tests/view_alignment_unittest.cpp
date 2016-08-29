#include <cstdint>

#include "gtest.h"

#include <Vector.hpp>

using namespace memory;

/// returns
///     true if ptr is aligned on n byte boundary
///     false otherwise
/// notes
///     - use convert to void* because the standard only guarentees
///       conversion to uintptr_t for void*
template <typename T>
bool test_alignment(const T* ptr, std::size_t n) {
    return std::uintptr_t( (const void*)(ptr) )%n == 0;
}

// check that const views work
TEST(view_alignment, indexed) {
    using coordinator = HostCoordinator<int, AlignedAllocator<int, 32>>;
    using array_type = Array<int, coordinator>;
    using view_type = typename array_type::view_type;

    array_type v(10);

    // take a view, which will have alignment of sizeof(int) -> 4
    // this should fail
    auto view = v(1, 10);
    EXPECT_TRUE(test_alignment(view.data(), coordinator::alignment()));
}
