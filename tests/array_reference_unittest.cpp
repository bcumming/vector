#include "gtest.h"

#include <Array.hpp>
#include <ArrayView.hpp>
#include <HostCoordinator.hpp>

TEST(ArrayReference, basics) {
    using namespace memory;

    using by_value = Array<double, HostCoordinator<double>>;
    using by_view  = ArrayView<double, HostCoordinator<double>>;
    using by_reference = typename by_view::array_reference_type;

    // create range by value of length 10
    by_value v1(10);
    EXPECT_EQ(v1.size(), 10);

    by_reference vr1 = v1(0,5);
    by_reference vr2 = v1(5,end);
    EXPECT_EQ(vr1.data(), v1.data());
    EXPECT_EQ(vr2.data(), v1.data()+5);
    v1(all) = 0.;
    vr1 = 1.;
    for(auto i=0; i<5; ++i)
        EXPECT_EQ(v1[i], 1.);
    vr2 = vr1;
    for(auto i=5; i<10; ++i)
        EXPECT_EQ(v1[i], 1.);

    // create a new array of lenght 10
    // and test copying it into the original array
    // via a reference
    by_value v2(10);
    EXPECT_EQ(v2.size(), 10);
    v2(all) = -10.;

    v1(all) = v2;
    EXPECT_NE(v1.data(), v2.data());

    for(auto i : v1.range())
        EXPECT_EQ(v1[i], v2[i]);
}

TEST(ArrayReference, metafunctions) {
    using namespace memory;

    using by_value = Array<double, HostCoordinator<double>>;
    using by_view  = ArrayView<double, HostCoordinator<double>>;
    using by_reference = typename by_view::array_reference_type;

    // create range by value of length 10
    // this should allocate memory of length 10*sizeof(T)
    by_value v1(10);
    EXPECT_EQ(v1.size(), 10);

    by_reference vr1 = v1(0,5);
    by_reference vr2 = v1(5,end);

    static_assert(impl::is_array_reference<decltype(vr1)>::value,
                  "incorrectly identified array reference");
    static_assert(impl::is_array_reference<decltype(v1(all))>::value,
                  "incorrectly identified array reference");
    static_assert(impl::is_array_reference<decltype(v1(0,1))>::value,
                  "incorrectly identified array reference");
}

