#include "gtest.h"

#include <vector.h>
#include <hostcoordinator.h>


// test that constructors work
TEST(host_vector, constructor) {
    using namespace memory;

    typedef host_coordinator<float> Coord;
    typedef range_by_value<float, Coord> RBV;
    typedef vector<RBV> host_vector;

    host_vector v2;
    host_vector v1(100);
    for(int i=0; i<v1.size(); ++i)
        v1[i] = float(i);

    //host_vector v3(v1(90, 100));
}

