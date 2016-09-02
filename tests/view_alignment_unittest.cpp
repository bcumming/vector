#include "gtest.h"

#include <Vector.hpp>
#include "util.hpp"

using namespace memory;

TEST(view_alignment, from_view) {
    using coordinator = HostCoordinator<int, AlignedAllocator<int, 32>>;
    using array_type = Array<int, coordinator>;
    using view_type = typename array_type::view_type;

    array_type v(10);

    // take a view that should have the correct alignment
    auto view = v(8, 10);
    EXPECT_TRUE(
        util::is_aligned(view.data(), coordinator::alignment())
    );

    // take views that have incorrect alignment
    EXPECT_THROW(auto _ = v(1,10), util::alignment_error); //  4 byte boundary
    EXPECT_THROW(auto _ = v(2,10), util::alignment_error); //  8 byte boundary
    EXPECT_THROW(auto _ = v(4,10), util::alignment_error); // 16 byte boundary
}

TEST(view_alignment, to_view) {
    using coordinator = HostCoordinator<int, AlignedAllocator<int, 16>>;
    using array_type = Array<int, coordinator>;
    using view_type = typename array_type::view_type;

    array_type source(10);
    array_type target(10);

    // copying the contents from one view to another should not have alignment issues
    for(auto i=0u; i<target.size()-2; ++i) {
        EXPECT_NO_THROW(target(i, i+2) = source(i, i+2));
    }
}
