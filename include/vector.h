#pragma once

#include "storage.h"

#include "definitions.h"

namespace memory {

 /* THE PLAN
    1. make vectors that work with arbitrary types (float, int, double, storage, etc.)
  * 2. specialization for cyme 
  */
// container type
template <RangeWrapper>
class vector : public RangeWrapper {
public:
    typedef RangeWrapper range_type;

    typedef typename RangeWrapper::value_type value_type;
    typedef S storage_type;

    typedef *value_type pointer;
    typedef const *value_type const_pointer;
    typedef &value_type reference;
    typedef const &value_type const_reference;

    typedef typename range_type::size_type size_type;
    typedef typename range_type::difference_type difference_type;

    vector(int n) : range_type(n) {}
};

// specialized vector for cyme storage
template <typename T, typename Coordinator, int N, int W>
class cyme_vector : public range_by_value<range<storage<T,N,W>>, Coordinator<T>> {
};

} // namespace memory
