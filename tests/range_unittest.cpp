#include <gtest/gtest.h>

#include <range.h>

// basic test that gtest is working and installed
TEST(range, reference) {
    const int N = 10;
    int arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    using namespace memory;

    Range<int> r1(arr, 10);
    EXPECT_EQ(10,r1.size());
    EXPECT_FALSE(r1.is_empty());

    ReferenceRange<int> r2 = r1(0, 5);
    EXPECT_EQ(5,r2.size());
    EXPECT_FALSE(r2.is_empty());

    ReferenceRange<int> r3 = r1(3, 3);
    EXPECT_EQ(0,r3.size());
    EXPECT_TRUE(r3.is_empty());
}

