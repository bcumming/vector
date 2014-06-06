#pragma once

#include <iostream>

#include "definitions.h"
#include "range_wrapper.h"
#include "host_coordinator.h"


namespace memory {

 /* THE PLAN
    1. this derived type serves as a "partially specialized template" workaround
       the same thing could be acheived more elegantly using c++11 features
  * 2. this vector type can then be used to create specific vector types specialized for memory spaces
  */
// container type
template <typename T, typename Coord>
class vector : public range_by_value<T, Coord> {
public:
    typedef range_by_value<T, Coord> range_wrapper_type;
    typedef typename get_reference_range<range_wrapper_type>::type reference_range;

    typedef typename range_wrapper_type::value_type value_type;
    typedef typename range_wrapper_type::coordinator_type coordinator_type;

    typedef value_type* pointer;
    typedef value_type const* const_pointer;
    typedef value_type& reference;
    typedef value_type const& const_reference;

    typedef typename range_wrapper_type::size_type size_type;
    typedef typename range_wrapper_type::difference_type difference_type;

    // default constructor : no memory will be allocated
    vector() : range_wrapper_type() {
        #ifdef MEMORY_DEBUG
        std::cout << "vector()" << std::endl;
        #endif
    }

    vector(int n) : range_wrapper_type(n) {
        #ifdef MEMORY_DEBUG
        std::cout << "vector(" << n << ")" << std::endl;
        #endif
    }

    // if reference type this will simply take a reference, otherwise copy out
    vector(reference_range const &rng) : range_wrapper_type(rng) {
        #ifdef MEMORY_DEBUG
        std::cout << "vector(reference_range)" << std::endl;
        #endif
    }

    // if reference type this will simply take a reference, otherwise copy out
    vector(range_wrapper_type const &rng) : range_wrapper_type(rng) {
        #ifdef MEMORY_DEBUG
        std::cout << "vector(value_range)" << std::endl;
        #endif
    }
};

// specialization for host vectors
template <typename T> using host_vector = vector<T, host_coordinator<T>>;

} // namespace memory
