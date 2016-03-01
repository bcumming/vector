#include "gtest.h"

#include <Array.hpp>
#include <ArrayView.hpp>
#include <HostCoordinator.hpp>

TEST(ArrayView, metafunctions) {
    using namespace memory;

    using by_value = Array<double, HostCoordinator<double>>;
    using by_view  = ArrayView<double, HostCoordinator<double>>;

    // create range by value of length 10
    // this should allocate memory of length 10*sizeof(T)
    by_value v1(10);
    EXPECT_EQ(v1.size(), 10);

    by_view  vv1 = v1(0,5);

    static_assert(impl::is_array_view<decltype(vv1)>::value,
            "is_array_view metafuntion clasified an ArrayView incorrectly");
}

#ifdef WITH_TBB
TEST(ArrayView, tbb_splitting) {
    using namespace memory;

    using by_value = Array<int, HostCoordinator<int>>;
    using by_view  = ArrayView<int, HostCoordinator<int>>;

    // create range by value of length 10
    // this should allocate memory of length 10*sizeof(T)
    by_value v(10);
    std::iota(v.begin(), v.end(), 0);

    static_assert(impl::is_array_view<by_view>::value,
            "is_array_view metafuntion clasified an ArrayView incorrectly");
    by_view  vv1 = v(memory::all);
    by_view  vv2(vv1, tbb::split());
    std::cout << vv1.size() << " + " << vv2.size()
              << " = " << vv1.size() + vv2.size() << std::endl;

    // assert that the two new views should sub-divide v
    EXPECT_EQ(vv1.size() + vv2.size(), v.size());
    EXPECT_EQ(vv1.data(), v.data());
    EXPECT_EQ(vv2.end(),  v.end());

    for(auto i : vv1.range()) {
        EXPECT_EQ(i, vv1[i]);
    }
    for(auto i : vv2.range()) {
        EXPECT_EQ(i+5, vv2[i]);
    }
}
#endif // WITH_TBB
