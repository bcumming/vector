#pragma once

#include <memory>
#include <algorithm>

#include "definitions.h"
#include "range.h"

namespace memory {
template <typename T, class Allocator=std::allocator<T> >
class host_coordinator {
public:
    typedef T value_type;

    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;

    typedef range<value_type> range_type;

    typedef typename types::size_type size_type;
    typedef typename types::difference_type difference_type;

    // metafunction for rebinding host_coordinator with another type
    template <typename Tother>
    struct rebind {
        typedef host_coordinator<Tother, Allocator> other;
    };

    range_type allocate(size_type n) {
        //todo make this work with alignment
        typename Allocator::template rebind<value_type>::other allocator;

        // only allocate memory if nonzero memory allocation has been requested
        pointer ptr = n>0 ? allocator.allocate(n) : 0;

        return range_type(ptr, n);
    }

    void free(range_type& rng) {
        //todo make this work with alignment
        typename Allocator::template rebind<value_type>::other allocator;

        if(rng.data())
            allocator.deallocate(rng.data(), rng.size());

        rng.reset();
    }

    // add guards to check that ranges are for value_type
    // there are two special cases
    //  1. R2 is a reference, in which case you do not free, and R2 must have the same length as R1.
    //  2. R2 is a base range, in which case free+realloc if R1.size() != R2.size(), before copying
    //      these special cases should be handled using specialization, with boost guards to ensure that R1 is valid
    template<typename R1, typename R2>
    void copy(const R1 &from, R2 &to) {
        // free memory associated with R2
        assert(from.size()==to.size());
        assert(!from.overlaps(to));

        std::copy(from.begin(), from.end(), to.begin());
    }
private:
};

} //namespace memory
