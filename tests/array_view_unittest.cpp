#include "gtest.h"

#include <Vector.hpp>

// check that const views work
TEST(array_view, constness) {
    using vector = memory::HostVector<int>;
    using view   = vector::view_type;
    using const_view  = vector::const_view_type;

    vector v(10);
    std::iota(v.begin(), v.end(), 0);

    view v_non_const = v;

    {
        const_view v_const(v);
        for(auto i : v.range()) {
            EXPECT_EQ(v[i], v_const[i]);
        }
    }
    {
        const_view v_const(v_non_const);
        for(auto i : v.range()) {
            EXPECT_EQ(v[i], v_const[i]);
        }
    }
    {
        const_view v_const(v(memory::all));
        for(auto i : v.range()) {
            EXPECT_EQ(v[i], v_const[i]);
        }
    }
}
