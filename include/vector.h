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
    typedef range_by_value<T, Coord> super;
    typedef typename get_reference_range<super>::type reference_range;

    typedef typename super::value_type value_type;
    typedef typename super::coordinator_type coordinator_type;

    typedef value_type* pointer;
    typedef value_type const* const_pointer;
    typedef value_type& reference;
    typedef value_type const& const_reference;

    typedef typename super::size_type size_type;
    typedef typename super::difference_type difference_type;

    // default constructor : no memory will be allocated
    vector() : super() {
        #ifdef MEMORY_DEBUG
        std::cout << "vector()" << std::endl;
        #endif
    }

    vector(int n) : super(n) {
        #ifdef MEMORY_DEBUG
        std::cout << "vector(" << n << ")" << std::endl;
        #endif
    }

    // if reference type this will simply take a reference, otherwise copy out
    vector(reference_range const &rng) : super(rng) {
        #ifdef MEMORY_DEBUG
        std::cout << "vector(reference_range)" << std::endl;
        #endif
    }

    // if reference type this will simply take a reference, otherwise copy out
    vector(super const &rng) : super(rng) {
        #ifdef MEMORY_DEBUG
        std::cout << "vector(value_range)" << std::endl;
        #endif
    }
};

// specialization for host vectors
template <typename T>
using host_vector = vector<T, host_coordinator<T>>;

#ifdef WITH_CUDA
// specialization for pinned vectors
// use a host_coordinator, because memory is  in the host memory space, and
// all of the helpers (copy, set, etc) are the same with and without page locked
// memory
template <typename T>
using pinned_vector = vector<T, host_coordinator<T, pinned_allocator<T>>>;

// specialization for device memory
//template <typename T> using device_vector = vector<T, device_coordinator<T>>;
#endif

} // namespace memory
