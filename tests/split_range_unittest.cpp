#include "gtest.h"

#include <algorithm>
#include <vector>

#include <host_coordinator.h>
#include <split_range.h>
#include <vector.h>

template <typename VEC>
void print(VEC const& v) {
    for(auto v_: v)
        std::cout << v_ << " ";
    std::cout << std::endl;
}

// test that constructors work
TEST(SplitRange, split) {
    using namespace memory;

    typedef HostVector<float> vector_type;
    typedef vector_type::view_type view_type;

    // length constructor
    vector_type vector(100);

    std::vector<view_type> splits;

    for(auto s : SplitRange(Range(0,100), 13)) 
        splits.push_back(vector(s));
}
