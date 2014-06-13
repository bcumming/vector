#include "gtest.h"

#include <allocator.h>
#include <host_coordinator.h>
#include <storage.h>

// verify that metafunction for memory alignment works
template <size_t s>
struct packer {
    char vals[s];
};

TEST(allocator, minimum_possible_alignment) {
    using namespace memory::impl;
    size_t tmp;
    tmp = minimum_possible_alignment< packer<1> >::value;
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<2> >::value;
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<3> >::value;
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<4> >::value;
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<5> >::value;
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<6> >::value;
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<7> >::value;
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<8> >::value;
    EXPECT_EQ(tmp,8);
    tmp = minimum_possible_alignment< packer<9> >::value;
    EXPECT_EQ(tmp,16);
    tmp = minimum_possible_alignment< packer<10> >::value;
    EXPECT_EQ(tmp,16);
    tmp = minimum_possible_alignment< packer<11> >::value;
    EXPECT_EQ(tmp,16);
    tmp = minimum_possible_alignment< packer<15> >::value;
    EXPECT_EQ(tmp,16);
    tmp = minimum_possible_alignment< packer<16> >::value;
    EXPECT_EQ(tmp,16);
    tmp = minimum_possible_alignment< packer<17> >::value;
    EXPECT_EQ(tmp,32);
    tmp = minimum_possible_alignment< packer<29> >::value;
    EXPECT_EQ(tmp,32);
    tmp = minimum_possible_alignment< packer<30> >::value;
    EXPECT_EQ(tmp,32);
    tmp = minimum_possible_alignment< packer<31> >::value;
    EXPECT_EQ(tmp,32);
    tmp = minimum_possible_alignment< packer<32> >::value;
    EXPECT_EQ(tmp,32);
    tmp = minimum_possible_alignment< packer<33> >::value;
    EXPECT_EQ(tmp,64);
    tmp = minimum_possible_alignment< packer<127> >::value;
    EXPECT_EQ(tmp,128);
    tmp = minimum_possible_alignment< packer<128> >::value;
    EXPECT_EQ(tmp,128);
    tmp = minimum_possible_alignment< packer<129> >::value;
    EXPECT_EQ(tmp,256);
    tmp = minimum_possible_alignment< packer<255> >::value;
    EXPECT_EQ(tmp,256);
    tmp = minimum_possible_alignment< packer<256> >::value;
    EXPECT_EQ(tmp,256);
    tmp = minimum_possible_alignment< packer<257> >::value;
    EXPECT_EQ(tmp,512);
}


