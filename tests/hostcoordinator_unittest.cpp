#include <gtest/gtest.h>

#include <hostcoordinator.h>
#include <storage.h>

TEST(hostcoordinator, initialization) {
    using namespace memory;

    typedef memory::Storage<float,  16, 4> StorageFloatAoSoA;

    typedef host_coordinator<int> intcoord_t;
    typedef host_coordinator<StorageFloatAoSoA> storagecoord_t;


    // verify that the correct type is used for internal storage
    ::testing::StaticAssertTypeEq<int,   intcoord_t::value_type>();
    ::testing::StaticAssertTypeEq<StorageFloatAoSoA, storagecoord_t::value_type>();
}

// test allocation of ranges
TEST(hostcoordinator, baserange_alloc_free) {
    using namespace memory;

    typedef host_coordinator<int> intcoord_t;
    intcoord_t coordinator;

    auto rng = coordinator.allocate(5);
    typedef decltype(rng) rng_t;

    // test that range is a base range
    EXPECT_TRUE(is_base_range<rng_t>::value);

    // test that range has correct storage type
    ::testing::StaticAssertTypeEq<int, rng_t::value_type >();

    // verify that the range has non-NULL pointer
    EXPECT_NE(rng_t::pointer(0), rng.data()) << "host_coordinator returned a NULL pointer when alloating a nonzero range";

    // verify that freeing works
    coordinator.free(rng);
    EXPECT_EQ(NULL, rng.data());
}
