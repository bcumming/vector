#include "gtest.h"

#include <array.h>

// check that ranges work
TEST(range, reference) {
    // get an array of 10 integers
    const int N = 10;
    int arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    using namespace memory;

    // check that a range can reference the array
    ArrayBase<int> r1(arr, 10);
    EXPECT_EQ(10,r1.size());        // correct size
    EXPECT_FALSE(r1.is_empty());    // should not be empty

    ArrayBase<int> r2 = r1(0, 5);
    EXPECT_EQ(5,r2.size());
    EXPECT_FALSE(r2.is_empty());

    ArrayBase<int> r3 = r1(3, 3);
    EXPECT_EQ(0,r3.size());         // should have length zero
    EXPECT_TRUE(r3.is_empty());     // should be empty
}

