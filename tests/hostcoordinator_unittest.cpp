#include <gtest/gtest.h>

#include <hostcoordinator.h>
#include <storage.h>

TEST(hostcoordinator, initialization) {
    using namespace memory;

    typedef memory::Storage<float,  16, 4> StorageFloatAoSoA;

    typedef host_coordinator<int> intcoord_t;
    typedef host_coordinator<StorageFloatAoSoA> storagecoord_t;
    //typedef typename host_coordinator<int>::value_type vt;

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

    intcoord_t coord;

    foo();

    // test that range is a base range
    EXPECT_TRUE(is_base_range<rng_t>::value);

    // test that range has correct storage type
    ::testing::StaticAssertTypeEq<int, rng_t::value_type >();

    // verify that the range has non-NULL pointer
    EXPECT_NE(rng_t::pointer(0), rng.data()) << "host_coordinator returned a NULL pointer when alloating a nonzero range";

    // verify that freeing works
    coordinator.free(rng);

    EXPECT_EQ(rng_t::pointer(0), rng.data());
    EXPECT_EQ(rng_t::size_type(0), rng.size());
}

// test allocation of ranges
TEST(hostcoordinator, refrange_alloc_free) {
    using namespace memory;

    typedef host_coordinator<float> floatcoord_t;
    floatcoord_t coordinator;

    auto rng = coordinator.allocate(5);
    typedef decltype(rng) rng_t;

    auto rrng = rng(all);
    typedef decltype(rrng) rrng_t;

    // test that range is a base range
    EXPECT_FALSE(is_base_range<rrng_t>::value);

    // test that range has correct storage type
    ::testing::StaticAssertTypeEq<float, rrng_t::value_type >();

    // verify that the range has non-NULL pointer
    EXPECT_NE(rrng_t::pointer(0), rrng.data()) << "host_coordinator returned a NULL pointer when alloating a nonzero range";

    EXPECT_EQ(rng.data(), rrng.data()) << "base(all) does not have the same pointer adress as base";

    // verify that freeing works
    coordinator.free(rng);

    EXPECT_EQ(rng_t::pointer(0),   rng.data());
    EXPECT_EQ(rng_t::size_type(0), rng.size());
}

