#include "gtest.h"

#include <range_wrapper.h>
#include <hostcoordinator.h>
#include <storage.h>

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
    rbv r1(10);
    EXPECT_EQ(r1.size(), 10);

    // create a copy
    rbv r2(r1);
    EXPECT_EQ(r2.size(), r1.size());
    EXPECT_NE(r2.data(), r1.data());

    rbr ref1(r1);
    EXPECT_EQ(ref1.size(), r1.size());
    EXPECT_EQ(ref1.data(), r1.data());
    rbv r3(ref1);
    EXPECT_EQ(r3.size(), r1.size());
}

TEST(range_wrapper,new_range_by_ref) {
    using namespace memory;

    typedef range_by_value<double, host_coordinator<double> > rbv;
    typedef range_by_reference<double, host_coordinator<double> > rbr;

    // create range by value of length 10
    // this should allocate memory of length 10*sizeof(T)
    rbv r1(10);
    rbr ref1(r1);

    EXPECT_EQ(ref1.size(), r1.size());
    EXPECT_EQ(ref1.data(), r1.data());
    rbv r2(ref1);
    EXPECT_EQ(r2.size(), r1.size());
    EXPECT_NE(r2.data(), r1.data());
}