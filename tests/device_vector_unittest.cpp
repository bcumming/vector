#include "gtest.h"

#include <vector.h>
#include <host_coordinator.h>
#include <device_coordinator.h>
#include <algorithm>

// test that constructors work
TEST(DeviceVector, constructor) {
    using namespace memory;

    // default constructor
    DeviceVector<float> v2;

    // length constructor
    DeviceVector<float> v1(100);
}