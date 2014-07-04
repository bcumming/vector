#pragma once

#include <memory>
#include <algorithm>

#include "definitions.h"
#include "range.h"
#include "allocator.h"

namespace memory {

// forward declare
template <typename T, class Allocator>
class host_coordinator;

namespace util {
    template <typename T, typename Allocator>
    struct type_printer<host_coordinator<T,Allocator>>{
        static std::string print() {
            std::stringstream str;
            str << "host_coordinator<" << type_printer<T>::print()
                << ", " << type_printer<Allocator>::print() << ">";
            return str.str();
        }
    };

    template <typename T, typename Allocator>
    struct pretty_printer<host_coordinator<T,Allocator>>{
        static std::string print(const host_coordinator<T,Allocator>& val) {
            std::stringstream str;
            str << type_printer<host_coordinator<T,Allocator>>::print();
            return str.str();
        }
    };
} // namespace util

template <typename T, class Allocator=aligned_allocator<T> >
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
        // todo make this work with alignment
        typename Allocator::template rebind<value_type>::other allocator;

        // only allocate memory if nonzero memory allocation has been requested
        pointer ptr = n>0 ? allocator.allocate(n) : nullptr;

        #ifndef NDEBUG
        std::cerr << util::type_printer<host_coordinator>::print()
                  << "::allocate(" << n << ") "
                  << (ptr==nullptr && n>0 ? " failure" : " success")
                  << std::endl;
        #endif

        return range_type(ptr, n);
    }

    void free(range_type& rng) {
        // todo make this work with alignment
        typename Allocator::template rebind<value_type>::other allocator;

        #ifndef NDEBUG
        std::cerr << util::type_printer<host_coordinator>::print()
                  << "::free()" << std::endl;
        #endif

        if(rng.data())
            allocator.deallocate(rng.data(), rng.size());

        rng.reset();
    }

    // copy memory from one range into another
    void copy(const range_type &from, range_type &to) {
        // free memory associated with R2
        assert(from.size()==to.size());
        assert(!from.overlaps(to));

        std::copy(from.begin(), from.end(), to.begin());
    }
};

} //namespace memory
