#pragma once

#include <memory>
#include <algorithm>

#include "definitions.h"
#include "Array.h"
#include "Allocator.h"

////////////////////////////////////////////////////////////////////////////////
namespace memory {

// forward declare
template <typename T, class Allocator>
class HostCoordinator;

////////////////////////////////////////////////////////////////////////////////
namespace util {
    template <typename T, typename Allocator>
    struct type_printer<HostCoordinator<T,Allocator>>{
        static std::string print() {
            std::stringstream str;
            str << "HostCoordinator<" << type_printer<T>::print()
                << ", " << type_printer<Allocator>::print() << ">";
            return str.str();
        }
    };

    template <typename T, typename Allocator>
    struct pretty_printer<HostCoordinator<T,Allocator>>{
        static std::string print(const HostCoordinator<T,Allocator>& val) {
            std::stringstream str;
            str << type_printer<HostCoordinator<T,Allocator>>::print();
            return str.str();
        }
    };
} // namespace util
////////////////////////////////////////////////////////////////////////////////

template <typename T, class Allocator=AlignedAllocator<T> >
class HostCoordinator {
public:
    typedef T value_type;

    typedef       value_type* pointer;
    typedef const value_type* const_pointer;
    typedef       value_type& reference;
    typedef const value_type& const_reference;

    typedef ArrayView<value_type, HostCoordinator> range_type;

    typedef typename types::size_type size_type;
    typedef typename types::difference_type difference_type;

    // metafunction for rebinding host_coordinator with another type
    template <typename Tother>
    struct rebind {
        typedef HostCoordinator<Tother, Allocator> other;
    };

    range_type allocate(size_type n) {
        // todo make this work with alignment
        typename Allocator::template rebind<value_type>::other allocator;

        // only allocate memory if nonzero memory allocation has been requested
        pointer ptr = n>0 ? allocator.allocate(n) : nullptr;

        #ifdef VERBOSE
        bool success = ptr;
        std::cerr << util::type_printer<HostCoordinator>::print()
                  << "::" + (success ? util::green("allocate") : util::red("allocate"))
                  << "(" << n*sizeof(value_type) << " bytes) @ " << ptr
                  << std::endl;
        #endif

        return range_type(ptr, n);
    }

    void free(range_type& rng) {
        // todo make this work with alignment
        typename Allocator::template rebind<value_type>::other allocator;

        #ifdef VERBOSE
        std::cerr << util::type_printer<HostCoordinator>::print()
                  << "::" + util::green("free")
                  << "(" << rng.size()*sizeof(value_type) << " bytes)"
                  << " @ " << rng.data()
                  << std::endl;
        #endif

        if(rng.data())
            allocator.deallocate(rng.data(), rng.size());

        rng.reset();
    }

    // copy memory from one range into another
    void copy(const range_type &from, range_type &to) {
        assert(from.size()==to.size());
        assert(!from.overlaps(to));

        std::copy(from.begin(), from.end(), to.begin());
    }

    // copy memory from one range into another
    void set(range_type &rng, value_type val) {
        std::fill(rng.begin(), rng.end(), val);
    }

    reference make_reference(value_type* p) {
        return *p;
    }

    const_reference make_reference(value_type const* p) const {
        return *p;
    }

};

} //namespace memory
////////////////////////////////////////////////////////////////////////////////
