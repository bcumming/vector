#pragma once

#include "definitions.h"
#include "array.h"
#include "allocator.h"

////////////////////////////////////////////////////////////////////////////////
namespace memory {

// forward declare
template <typename T, class Allocator>
class DeviceCoordinator;

////////////////////////////////////////////////////////////////////////////////
namespace util {
    template <typename T, typename Allocator>
    struct type_printer<DeviceCoordinator<T,Allocator>>{
        static std::string print() {
            std::stringstream str;
            str << "DeviceCoordinator<" << type_printer<T>::print()
                << ", " << type_printer<Allocator>::print() << ">";
            return str.str();
        }
    };

    template <typename T, typename Allocator>
    struct pretty_printer<DeviceCoordinator<T,Allocator>>{
        static std::string print(const DeviceCoordinator<T,Allocator>& val) {
            std::stringstream str;
            str << type_printer<DeviceCoordinator<T,Allocator>>::print();
            return str.str();
        }
    };
} // namespace util
////////////////////////////////////////////////////////////////////////////////

template <typename T>
class ConstDeviceReference {
public:
    typedef T  value_type;
    typedef T* pointer;

    ConstDeviceReference(pointer p) : pointer_(p) {}

    operator T() const {
        T tmp;
        cudaMemcpy(&tmp, pointer_, sizeof(T), cudaMemcpyDeviceToHost );
        return T(tmp);
    }

protected:
    pointer pointer_;
};

template <typename T>
class DeviceReference {
public:
    typedef T  value_type;
    typedef T* pointer;

    DeviceReference(pointer p) : pointer_(p) {}

    DeviceReference& operator = (const T& value) {
        cudaMemcpy(pointer_, &value, sizeof(T), cudaMemcpyHostToDevice );
        return *this;
    }

    operator T() const {
        T tmp;
        cudaMemcpy(&tmp, pointer_, sizeof(T), cudaMemcpyDeviceToHost );
        return T(tmp);
    }

private:
    pointer pointer_;
};


template <typename T, class Allocator_=CudaAllocator<T> >
class DeviceCoordinator {
public:
    typedef T value_type;
    typedef typename Allocator_::template rebind<value_type>::other Allocator;

    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef DeviceReference<T> reference;
    typedef ConstDeviceReference<T> const_reference;

    typedef ArrayBase<value_type> range_type;

    typedef typename types::size_type size_type;
    typedef typename types::difference_type difference_type;

    // metafunction for rebinding host_coordinator with another type
    template <typename Tother>
    struct rebind {
        typedef DeviceCoordinator<Tother, Allocator> other;
    };

    range_type allocate(size_type n) {
        Allocator allocator;

        // only allocate memory if nonzero memory allocation has been requested
        pointer ptr = n>0 ? allocator.allocate(n) : 0;

        #ifndef NDEBUG
        std::cerr << util::type_printer<DeviceCoordinator>::print()
                  << "::allocate(" << n << ") "
                  << (ptr==nullptr && n>0 ? " failure" : " success")
                  << std::endl;
        #endif

        return range_type(ptr, n);
    }

    void free(range_type& rng) {
        Allocator allocator;

        if(rng.data())
            allocator.deallocate(rng.data(), rng.size());

        #ifndef NDEBUG
        std::cerr << util::type_printer<DeviceCoordinator>::print()
                  << "::free()" << std::endl;
        #endif

        rng.reset();
    }

    // copy memory from one gpu range to another
    void copy(const range_type &from, range_type &to) {
        assert(from.size()==to.size());
        assert(!from.overlaps(to));

        cudaError_t status = cudaMemcpy(
                reinterpret_cast<void*>(to.begin()),
                reinterpret_cast<void*>(from.begin()),
                from.size()*sizeof(value_type),
                cudaMemcpyDeviceToDevice
        );
    }

    reference make_reference(value_type* p) {
        return reference(p);
    }

    const_reference make_reference(value_type const* p) const {
        return const_reference(p);
    }

    // copy memory from gpu range to host range
    /*
    template <typename >
    void copy(const range_type &from, range_type &tto) {
        // free memory associated with R2
        assert(from.size()==to.size());
        assert(!from.overlaps(to));

        cudaError_t status = cudaMemcpy(
                reinterpret_cast<void*>(to.begin()),
                reinterpret_cast<void*>(from.begin()),
                from.size()*sizeof(value_type),
                cudaMemcpyDeviceToDevice
        );
    }
    */

    // fill memory
    //void fill(range_type &rng, const T& value) {
    //}
};

} // namespace memory
////////////////////////////////////////////////////////////////////////////////

