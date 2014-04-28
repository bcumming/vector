#pragma once

#include <memory>
#include <algorithm>

#include "definitions.h"
#include "range.h"

namespace memory {
template <typename T, class Allocator=std::allocator<T> >
class host_coordinator {
public:
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;

    typedef Range<T> range_type;
    typedef ReferenceRange<T> reference_range_type;

    typedef typename types::size_type size_type;
    typedef typename types::difference_type difference_type;

    range_type allocate(size_type n) {
        //todo make this work with alignment
        typename Allocator::template rebind<T>::other allocator;

        // only allocate memory if nonzero memory allocation has been requested
        pointer ptr = n>0 ? allocator.allocate(n) : 0;

        return range_type(ptr, n);
    }

    void free(range_type& rng) {
        //todo make this work with alignment
        typename Allocator::template rebind<T>::other allocator;

        if(rng.data())
            allocator.free(rng.data());

        rng.reset;
    }

    // add guards to check that ranges are for T
    // there are two special cases
    //  1. R2 is a reference, in which case you do not free, and R2 must have the same length as R1.
    //  2. R2 is a base range, in which case free+realloc if R1.size() != R2.size(), before copying
    //      these special cases should be handled using specialization, with boost guards to ensure that R1 is valid
    template<typename R1, typename R2>
    void copy(const R1 &from, R2 &to) {
        // free memory associated with R2
        // if R2 is a reference range, this will do nothing
        this->free(to);

        this->allocate(to, from.size());

        std::copy(from.begin(), from.end(), to.begin());
    }

    // do nothing for reference ranges, because a reference range does not own the
    // memory it refers to
    void free(reference_range_type& rng) {}
private:
};

} //namespace memory
