#include "gtest.h"

#include <Array.h>

TEST(ArrayBase, indexes) {
    // get an array of 10 integers
    const int N = 10;
    int arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    using namespace memory;

    // check that an array type can reference arr
    ArrayBase<int> a1(arr, 10);
    EXPECT_EQ(10,a1.size());        // correct size
    EXPECT_FALSE(a1.is_empty());    // should not be empty

    ArrayBase<int> a2 = a1(0, 5);
    EXPECT_EQ(5,a2.size());
    EXPECT_FALSE(a2.is_empty());

    ArrayBase<int> a3 = a1(3, 3);
    EXPECT_EQ(0,a3.size());         // should have length zero
    EXPECT_TRUE(a3.is_empty());     // should be empty
}


TEST(ArrayBase, ranges) {
    // get an array of 10 integers
    const int N = 10;
    int arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    using namespace memory;

    // check that an array type can reference arr
    ArrayBase<int> a1(arr, 10);
    EXPECT_EQ(10,a1.size());        // correct size
    EXPECT_FALSE(a1.is_empty());    // should not be empty

    ArrayBase<int> a2 = a1(Range(0,5));
    EXPECT_EQ(5,a2.size());
    EXPECT_FALSE(a2.is_empty());

    ArrayBase<int> a3 = a1(Range(3,3));
    EXPECT_EQ(0,a3.size());         // should have length zero
    EXPECT_TRUE(a3.is_empty());     // should be empty
}

TEST(ArrayBase, ranges_and_indexes) {
    // get an array of 10 integers
    const int N = 10;
    int arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    using namespace memory;

    // check that an array type can reference arr
    ArrayBase<int> a1(arr, 10);
    EXPECT_EQ(10,a1.size());        // correct size
    EXPECT_FALSE(a1.is_empty());    // should not be empty

    ArrayBase<int> ar = a1(Range(0,5));
    ArrayBase<int> ai = a1(0, 5);
    EXPECT_TRUE(ai==ar);    // should refer to the same memory
    EXPECT_TRUE(a1(Range(0,5))==a1(0,5)); // should refer to the same memory

    // check that equality returns false correctly
    EXPECT_FALSE(a1(Range(0,5))==a1(1,6)); // different pointer
    EXPECT_FALSE(a1(Range(0,5))==a1(0,6)); // different length
    EXPECT_FALSE(a1(Range(0,5))==a1(8,10)); // different pointer and length
}
