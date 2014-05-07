#pragma once

#pragma <iostream>


#include "definitions.h"
#include "range_wrapper.h"

namespace memory {

 /* THE PLAN
    1. make vectors that work with arbitrary types (float, int, double, storage, etc.)
  * 2. specialization for cyme 
  */
// container type
template <typename RangeWrapper>
class vector : public RangeWrapper {
public:
    typedef RangeWrapper range_wrapper_type;
    typedef typename get_reference_range<RangeWrapper>::type reference_range;

    typedef typename RangeWrapper::value_type value_type;

    typedef value_type* pointer;
    typedef value_type const* const_pointer;
    typedef value_type& reference;
    typedef value_type const& const_reference;

    typedef typename range_wrapper_type::size_type size_type;
    typedef typename range_wrapper_type::difference_type difference_type;

    vector(int n) : range_wrapper_type(n) {
        std::cout << "vector(int)" << std::endl;
    }
    vector() : range_wrapper_type() {
        std::cout << "vector()" << std::endl;
    }

    // if reference type this will simply take a reference, otherwise copy out
    vector(reference_range const &rng) : range_wrapper_type(rng) {
        std::cout << "vector(reference_range)" << std::endl;
    }

    // if reference type this will simply take a reference, otherwise copy out
    vector(range_wrapper_type const &rng) : range_wrapper_type(rng) {
        std::cout << "vector(range_wrapper_type)" << std::endl;
    }

    reference_range operator()(int from, int to) {
        return range_wrapper_type(from, to);
    }
};

/*
// specialized vector for cyme storage
template <typename T, typename Coordinator, int N, int W>
class cyme_vector : public range_by_value<range<storage<T,N,W>>, Coordinator<T>> {
};
*/

} // namespace memory
