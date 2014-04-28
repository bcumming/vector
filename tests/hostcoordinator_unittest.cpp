#include <gtest/gtest.h>

#include <hostcoordinator.h>
#include <storage.h>

template <typename R>
void print_range(const R& rng) {
    for(auto v: rng)
        std::cout << v << " ";
    std::cout << std::endl;
}

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

TEST(hostcoordinator, copy) {
    using namespace memory;

    const int N = 20;

    typedef host_coordinator<int> intcoord_t;
    intcoord_t coordinator;

    auto rng = coordinator.allocate(N);
    int i=0;
    for(auto &v: rng)
        v = i++;

    // this test produces a warning due to threading. 
    //ASSERT_DEATH(rng(0,N/2+1) = rng(N/2,end), "");
    rng(0,N/2) = rng(N/2,end);
    for(auto i=0; i<N/2; i++)
        EXPECT_EQ(rng[i], rng[i+N/2]);
    //print_range(rng);

    // create a new range of the same length, and initialize to new values
    auto rng2 = coordinator.allocate(N);
    i=0;
    for(auto &v: rng2)
        v = (i+=2);

    // take a reference range to the original range
    auto rrng = rng(all);

    rrng(all) = rng2(all);
    print_range(rng);
}
