#include "gtest.h"

#include <range_wrapper.h>
#include <hostcoordinator.h>
#include <storage.h>

// helper function for outputting a range
template <typename R>
void print_range(const R& rng) {
    for(auto v: rng)
        std::cout << v << " ";
    std::cout << std::endl;
}

// verify that metafunctions for checking range wrappers work
TEST(range_wrappers, is_range_wrapper) {
    using namespace memory;

    typedef range_by_value<double, host_coordinator<double> > rbv;
    typedef range_by_reference<double, host_coordinator<double> > rbr;

    static_assert(
        is_range_by_value<rbv>::value,
        "is_range_by_value incorrectly returned false for range_by_value" );

    static_assert(
        !is_range_by_reference<rbv>::value,
        "is_range_by_reference incorrectly returned true for range_by_value" );

    static_assert(
        !is_range_by_value<rbr>::value,
        "is_range_by_value incorrectly returned true for range_by_reference" );

    static_assert(
        is_range_by_reference<rbr>::value,
        "is_range_by_reference incorrectly returned false for range_by_reference" );

    static_assert(
        is_range_wrapper<rbv>::value,
        "is_range_wrapper incorrectly returned false for range_by_value" );

    static_assert(
        is_range_wrapper<rbr>::value,
        "is_range_wrapper incorrectly returned false for range_by_reference" );

    static_assert(
        !is_range_by_value<int>::value,
        "is_range_by_value returns true for type other than range_by_value" );

    static_assert(
        !is_range_by_reference<int>::value,
        "is_range_by_reference returns true for type other than range_by_value" );
}

TEST(range_wrapper,new_range_by_value) {
    using namespace memory;

    typedef range_by_value<double, host_coordinator<double> > rbv;
    typedef range_by_reference<double, host_coordinator<double> > rbr;

    // create range by value of length 10
    // this should allocate memory of length 10*sizeof(T)
    rbv v1(10);
    EXPECT_EQ(v1.size(), 10);

    // create a copy
    rbv v2(v1);
    EXPECT_EQ(v2.size(), v1.size());
    EXPECT_NE(v2.data(), v1.data());

    rbr r1(v1);
    EXPECT_EQ(r1.size(), v1.size());
    EXPECT_EQ(r1.data(), v1.data());
    rbv v3(r1);
    EXPECT_EQ(v3.size(), r1.size());
    EXPECT_NE(v3.data(), r1.data());
}

TEST(range_wrapper,new_range_by_ref) {
    using namespace memory;

    typedef range_by_value<double, host_coordinator<double> > rbv;
    typedef range_by_reference<double, host_coordinator<double> > rbr;

    // create range by value of length 10
    // this should allocate memory of length 10*sizeof(T)
    rbv v1(10);
    rbr r1(v1);

    EXPECT_EQ(r1.size(), v1.size());
    EXPECT_EQ(r1.data(), v1.data());
    rbv v2(r1);
    EXPECT_EQ(v2.size(), v1.size());
    EXPECT_NE(v2.data(), v1.data());
}

TEST(range_wrapper,sub_ranges) {
    using namespace memory;

    typedef range_by_value<double, host_coordinator<double> > rbv;
    typedef range_by_reference<double, host_coordinator<double> > rbr;

    // check that reference range from a subrange works
    rbv v1(10);
    auto a = v1(all);
    //rbr r1(v1(all));
    for(int i=0; i<10; i++)
        v1[i] = double(i);

/*
    EXPECT_EQ(r1.size(), 5);
    EXPECT_EQ(v1.data()+5, r1.data());
    print_range(v1);
    print_range(r1);

    rbv v2(v1);
    EXPECT_NE(v1.data(), v2.data());
    print_range(v2);
*/
}
