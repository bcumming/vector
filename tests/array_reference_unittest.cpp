#include "gtest.h"

#include <Array.hpp>
#include <ArrayView.hpp>
#include <HostCoordinator.hpp>

TEST(ArrayReference, indexes) {
    using namespace memory;

    using by_value = Array<double, HostCoordinator<double>>;
    using by_view  = ArrayView<double, HostCoordinator<double>>;
    using by_reference = typename by_view::array_reference_type;

    // create range by value of length 10
    // this should allocate memory of length 10*sizeof(T)
    by_value v1(10);
    by_value v2(10);
    EXPECT_EQ(v1.size(), 10);
    v2(all) = -10;

    std::cout << util::red("------- making vr1 ----------") << std::endl;
    by_reference vr1 = v1(0,5);
    std::cout << util::red("------- making vr2 ----------") << std::endl;
    by_reference vr2 = v1(5,end);
    std::cout << util::red("------- setting v1(all) = 0. --------------") << std::endl;
    v1(all) = 0.;
    std::cout << util::red("------- setting vr1 = 1. --------------") << std::endl;
    vr1 = 1.;
    std::cout << util::red("------- vr2 = vr1 --------------") << std::endl;
    vr2 = vr1;
    for(auto i: v1)
        std::cout << i << " ";
    std::cout << std::endl;
    std::cout << util::red("---------------------") << std::endl;
    v1(all) = v2;

    for(auto i: v1)
        std::cout << i << " ";
    std::cout << std::endl;
    EXPECT_TRUE(true);     // should be empty
}

