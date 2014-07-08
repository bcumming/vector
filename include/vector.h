#pragma once

#include <iostream>

#include "definitions.h"
#include "array.h"
#include "host_coordinator.h"


namespace memory {

 /* THE PLAN
    1. this derived type serves as a "partially specialized template" workaround
       the same thing could be acheived more elegantly using c++11 features
  * 2. this vector type can then be used to create specific vector types specialized for memory spaces
  */
// container type
template <typename T, typename Coord>
class Vector : public Array<T, Coord> {
public:
    typedef Array<T, Coord> base;
    typedef typename get_reference_range<base>::type reference_range;

    typedef typename base::value_type value_type;
    typedef typename base::coordinator_type coordinator_type;

    typedef value_type* pointer;
    typedef value_type const* const_pointer;
    typedef value_type& reference;
    typedef value_type const& const_reference;

    typedef typename base::size_type size_type;
    typedef typename base::difference_type difference_type;

    // default constructor : no memory will be allocated
    Vector() : base() {}

    Vector(int n) : base(n) {}

    // if reference type this will simply take a reference, otherwise copy out
    Vector(reference_range const &rng) : base(rng) {}

    // if reference type this will simply take a reference, otherwise copy out
    Vector(base const &rng) : base(rng) {}
};

// specialization for host vectors
template <typename T>
using HostVector = Vector<T, HostCoordinator<T>>;

#ifdef WITH_CUDA
// specialization for pinned vectors
// use a host_coordinator, because memory is  in the host memory space, and
// all of the helpers (copy, set, etc) are the same with and without page locked
// memory
template <typename T>
using pinned_vector = Vector<T, HostCoordinator<T, pinned_allocator<T>>>;

// specialization for device memory
template <typename T>
using device_vector = Vector<T, HostCoordinator<T, cuda_allocator<T>>>;
#endif

} // namespace memory
