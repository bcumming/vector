#include "gtest.h"

#include <algorithm>

#include <Vector.h>
#include <HostCoordinator.h>

template <typename VEC>
void print(VEC const& v) {
    for(auto v_: v)
        std::cout << v_ << " ";
    std::cout << std::endl;
}

// test that constructors work
TEST(HostVector, constructor) {
    using namespace memory;

    // default constructor
    HostVector<float> v2;

    // length constructor
    HostVector<float> v1(100);

    // initialize values as monotone sequence
    for(int i=0; i<v1.size(); ++i)
        v1[i] = float(i);

    // initialize new HostVector from a subrange
    HostVector<float> v3(v1(90, 100));

    // reset values in range
    for(auto &v : v1(90, 100))
        v = float(-1.0);

    // check that v3 has values originally copied over from v1
    for(int i=0; i<10; i++)
        EXPECT_EQ(float(i+90), v3[i]);

    for(int i=90; i<100; i++)
        EXPECT_EQ(float(-1), v1[i]);
}

// test that copy constructors work
TEST(HostVector, copy_constructor) {
    using namespace memory;

    // length constructor
    HostVector<float> v1(100);

    // initialize values as monotone sequence
    for(int i=0; i<v1.size(); ++i)
        v1[i] = float(i);

    // copy constructor
    //HostVector<float> v2;
    //v2 = v1;
    HostVector<float> v2 = v1;

    // ensure that new memory was allocated
    EXPECT_NE(v2.data(), v1.data());

    // check that v3 has values originally copied over from v1
    for(int i=0; i<100; i++)
        EXPECT_EQ(v1[i], v2[i]);
}

// test that move constructors work
TEST(HostVector, move_constructor) {
    using namespace memory;

    // move constructor
    HostVector<float> v1 = HostVector<float>(100);

    for(auto i : v1.range())
        v1[i] = i;
}

/*
// test that iterators and ranges work
TEST(HostVector, iterators_and_ranges) {
    using namespace memory;

    // length constructor
    HostVector<float> v1(100);

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
    std::for_each(v1.begin(), v1.end(), [] (float& val) {val+=1;}); // add 1 to every value in v1
    {
        float sum = 0;
        for(auto val : v1)
            sum+=val;
        EXPECT_EQ(float(4*100), sum);
    }
}*/
