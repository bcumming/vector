#include "gtest.h"

#include <vector.h>
#include <host_coordinator.h>
#include <device_coordinator.h>
#include <algorithm>

template <typename VEC>
void print(VEC const& v) {
    for(auto v_: v)
        std::cout << v_ << " ";
    std::cout << std::endl;
}

#ifdef WITH_CUDA
// test that constructors work
TEST(pinned_vector, constructor) {
    using namespace memory;

    // default constructor
    pinned_vector<float> v2;

    // length constructor
    pinned_vector<float> v1(100);

    // initialize values as monotone sequence
    for(int i=0; i<v1.size(); ++i)
        v1[i] = float(i);

    // initialize new pinned_vector from a subrange
    pinned_vector<float> v3(v1(90, 100));

    // reset values in range
    for(auto &v : v1(90, 100))
        v = float(-1.0);

    // check that v3 has values originally copied over from v1
    for(int i=0; i<10; i++)
        EXPECT_EQ(float(i+90), v3[i]);

    for(int i=90; i<100; i++)
        EXPECT_EQ(float(-1), v1[i]);

    std::cerr << "-- CLEANP --\n";
}

// test that constructors work
TEST(pinned_vector, iterators_and_ranges) {
    using namespace memory;

    // length constructor
    pinned_vector<float> v1(100);

    // check that begin()/end() iterators work
    for(auto it=v1.begin(); it<v1.end(); ++it)
        *it = float(3.0);

    // check that range based for loop works
    for(auto &val : v1)
        val = float(3.0);
    {
        float sum = 0;
        // check it works for const
        for(auto val : v1)
            sum+=val;
        EXPECT_EQ(float(3*100), sum);
    }

    // check that std::for_each works
    // add 1 to every value in v1
    std::for_each(v1.begin(), v1.end(), [] (float& val) {val+=1;});
    {
        float sum = 0;
        for(auto val : v1)
            sum+=val;
        EXPECT_EQ(float(4*100), sum);
    }
}

#endif