#include "gtest.h"

#include <device_coordinator.h>
#include <host_coordinator.h>
#include <storage.h>

// verify that type members set correctly
TEST(DeviceCoordinator, type_members) {
    using namespace memory;

    typedef memory::Storage<float,  16, 4> StorageFloatAoSoA;

    typedef DeviceCoordinator<int> intcoord_t;
    typedef DeviceCoordinator<StorageFloatAoSoA> storagecoord_t;

    // verify that the correct type is used for internal storage
    ::testing::StaticAssertTypeEq<int,   intcoord_t::value_type>();
    ::testing::StaticAssertTypeEq<StorageFloatAoSoA,
                                  storagecoord_t::value_type>();
}

// verify that rebinding works
TEST(DeviceCoordinator, rebind) {
    using namespace memory;

    typedef DeviceCoordinator<int> intcoord_t;
    typedef typename intcoord_t::rebind<double>::other doublecoord_t;

    // verify that the correct type is used for internal storage
    ::testing::StaticAssertTypeEq<double,doublecoord_t::value_type>();
}

// test allocation of base arrays using host_coordinator
TEST(DeviceCoordinator, arraybase_alloc_free) {
    using namespace memory;

    typedef DeviceCoordinator<int> intcoord_t;
    intcoord_t coordinator;

    auto array = coordinator.allocate(5);
    typedef decltype(array) arr_t;

    intcoord_t coord;

    // test that array is a base array
    EXPECT_TRUE(is_array_by_reference<arr_t>::value);

    // test that array has correct storage type
    ::testing::StaticAssertTypeEq<int, arr_t::value_type >();

    // verify that the array has non-NULL pointer
    EXPECT_NE(arr_t::pointer(0), array.data())
        << "DeviceCoordinator returned a NULL pointer when allocating a nonzero array";

    // verify that freeing works
    coordinator.free(array);

    EXPECT_EQ(arr_t::pointer(0), array.data());
    EXPECT_EQ(arr_t::size_type(0), array.size());
}

// test allocation of reference arrays
TEST(DeviceCoordinator, refarray_alloc_free) {
    using namespace memory;

    typedef DeviceCoordinator<float> floatcoord_t;
    floatcoord_t coordinator;

    auto array = coordinator.allocate(5);
    typedef decltype(array) rng_t;

    auto ref_array = array(all);
    typedef decltype(ref_array) rrng_t;

    // test that array has correct storage type
    ::testing::StaticAssertTypeEq<float, rrng_t::value_type >();

    // verify that the array has non-NULL pointer
    EXPECT_NE(rrng_t::pointer(0), ref_array.data())
        << "DeviceCoordinator returned a NULL pointer when allocating a nonzero array";

    EXPECT_EQ(array.data(), ref_array.data())
        << "base(all) does not have the same pointer address as base";

    // verify that freeing works
    coordinator.free(array);

    EXPECT_EQ(rng_t::pointer(0),   array.data());
    EXPECT_EQ(rng_t::size_type(0), array.size());
}

// test that DeviceCoordinator can correctly detect overlap between arrays
TEST(DeviceCoordinator, overlap) {
    using namespace memory;

    const int N = 20;

    typedef DeviceCoordinator<int> intcoord_t;
    intcoord_t coordinator;

    auto array = coordinator.allocate(N);
    auto array_other = coordinator.allocate(N);
    EXPECT_FALSE(array.overlaps(array_other));
    EXPECT_FALSE(array(0,10).overlaps(array(10,end)));
    EXPECT_FALSE(array(10,end).overlaps(array(0,10)));

    EXPECT_TRUE(array.overlaps(array));
    EXPECT_TRUE(array(all).overlaps(array));
    EXPECT_TRUE(array.overlaps(array(all)));
    EXPECT_TRUE(array(all).overlaps(array(all)));
    EXPECT_TRUE(array(0,11).overlaps(array(10,end)));
    EXPECT_TRUE(array(10,end).overlaps(array(0,11)));
}

// test copy from host to device memory works
TEST(DeviceCoordinator, host_to_device_copy) {
    using namespace memory;

    const int N = 100;

    {
        typedef int T;
        typedef DeviceCoordinator<T> dc_t;
        typedef HostCoordinator<T>   hc_t;
        typedef ArrayView<T, dc_t> da_t;
        typedef ArrayView<T, hc_t> ha_t;

        // allocate array on host and device
        ha_t host_array(hc_t().allocate(N));
        da_t device_array(dc_t().allocate(N));

        // initialize host memory to linear sequence of integers
        for(auto i: Range(0,N))
            host_array[i] = T(i);

        // copy host array to device array
        dc_t().copy(host_array, device_array);

        // check that host and device values are the same
        for(auto i: Range(0,N))
            EXPECT_EQ( host_array[i], T(device_array[i]) );
    }

    {
        typedef double T;
        typedef DeviceCoordinator<T> dc_t;
        typedef HostCoordinator<T>   hc_t;
        typedef ArrayView<T, dc_t> da_t;
        typedef ArrayView<T, hc_t> ha_t;

        // allocate array on host and device
        ha_t host_array(hc_t().allocate(N));
        da_t device_array(dc_t().allocate(N));

        // initialize host memory to linear sequence of integers
        for(auto i: Range(0,N))
            host_array[i] = T(i);

        // copy host array to device array
        dc_t().copy(host_array, device_array);

        // check that host and device values are the same
        for(auto i: Range(0,N))
            EXPECT_EQ( host_array[i], T(device_array[i]) );
    }
}
