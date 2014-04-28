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

/*
#include <cstdlib>
#include <vector>
#include <iostream>

#include "range.h"

template<typename R>
void print_range(const R& rng) {
    for(int i=0; i<rng.size(); i++)
        std::cout << rng[i] << " ";
    std::cout << std::endl;
}

template<typename R>
void print_range_stats(const R& rng) {
    std::cout << "range has size " << rng.size() << " and is " << (rng.is_empty() ? "not " : "") << "empty" << std::endl;
}

int main(void) {
    std::vector<double> v(10);
    for(int i=0; i<10; i++)
        v[i] = double(i);

    Range<double,false> r(&(v[0]), 10);
    Range<double,true> r1 = r(4,end);
    Range<double,true> r2 = r1(1,3);

    print_range(r);
    print_range(r1);
    print_range(r2);

    foo(end);
    foo(all);

    return 0;
}
*/
