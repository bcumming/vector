#pragma once

#include <iostream>

#include "definitions.h"
#include "array.h"
#include "host_coordinator.h"


namespace memory {

 /* THE PLAN
  * - I just went ahead and used C++11
    1. this derived type serves as a "partially specialized template" workaround
       the same thing could be achieved more elegantly using c++11 features
  * 2. this vector type can then be used to create specific vector types specialized for memory spaces
  */
// container type
/*
template <typename T, typename Coord>
class Vector : public Array<T, Coord> {
public:
    typedef Array<T, Coord> base;
    typedef typename get_view<base>::type view_type;

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
    Vector(view_type const &v) : base(v) {}

    // if reference type this will simply take a reference, otherwise copy out
    Vector(base const &v) : base(v) {}
};
*/

// specialization for host vectors
template <typename T>
using HostVector = Array<T, HostCoordinator<T>>;
//using HostVector = Vector<T, HostCoordinator<T>>;

#ifdef WITH_CUDA
// specialization for pinned vectors. Use a host_coordinator, because memory is
// in the host memory space, and all of the helpers (copy, set, etc) are the
// same with and without page locked memory
template <typename T>
using PinnedVector = Array<T, HostCoordinator<T, PinnedAllocator<T>>>;

// specialization for device memory
template <typename T>
using DeviceVector = Array<T, HostCoordinator<T, CudaAllocator<T>>>;
#endif

} // namespace memory
