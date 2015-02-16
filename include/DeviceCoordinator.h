#pragma once

#include "Allocator.h"
#include "Array.h"
#include "definitions.h"
#include "Event.h"
#include "CudaEvent.h"

////////////////////////////////////////////////////////////////////////////////
namespace memory {

// forward declare
template <typename T, class Allocator>
class DeviceCoordinator;

//template <typename T, class Allocator=AlignedAllocator<T> >
template <typename T, class Allocator>
class HostCoordinator;

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
    using value_type = T;
    using pointer = *value_type;

    ConstDeviceReference(pointer p) : pointer_(p) {}

    operator T() const {
        T tmp;
        cudaMemcpy(&tmp, pointer_, sizeof(T), cudaMemcpyDeviceToHost);
        return T(tmp);
    }

protected:
    pointer pointer_;
};

template <typename T>
class DeviceReference {
public:
    using value_type = T;
    using pointer = *value_type;

    DeviceReference(pointer p) : pointer_(p) {}

    DeviceReference& operator = (const T& value) {
        cudaMemcpy(pointer_, &value, sizeof(T), cudaMemcpyHostToDevice);
        return *this;
    }

    operator T() const {
        T tmp;
        cudaMemcpy(&tmp, pointer_, sizeof(T), cudaMemcpyDeviceToHost);
        return T(tmp);
    }

private:
    pointer pointer_;
};


template <typename T, class Allocator_=CudaAllocator<T> >
class DeviceCoordinator {
public:
    using value_type = T;
    using Allocator = typename Allocator_::template rebind<value_type>::other;

    using pointer       = *value_type;
    using const_pointer = const value_type*;
    using reference       = DeviceReference<value_type>;
    using const_reference = ConstDeviceReference<value_type>;

    using array_type = ArrayView<value_type, DeviceCoordinator>;

    using size_type       = typename types::size_type;
    using difference_type = typename types::difference_type;

    // metafunction for rebinding host_coordinator with another type
    template <typename Tother>
    struct rebind {
        typedef DeviceCoordinator<Tother, Allocator> other;
    };

    //template <typename Tother>
    //using rebind = DeviceCoordinator<Tother, Allocator>;

    array_type allocate(size_type n) {
        Allocator allocator;

        // only allocate memory if nonzero memory allocation has been requested
        pointer ptr = n>0 ? allocator.allocate(n) : 0;

        #ifdef VERBOSE
        std::cerr << util::type_printer<DeviceCoordinator>::print()
                  << "::allocate(" << n << ") "
                  << (ptr==nullptr && n>0 ? " failure" : " success")
                  << std::endl;
        #endif

        return array_type(ptr, n);
    }

    void free(array_type& rng) {
        Allocator allocator;

        if(rng.data())
            allocator.deallocate(rng.data(), rng.size());

        #ifdef VERBOSE
        std::cerr << util::type_printer<DeviceCoordinator>::print()
                  << "::free()" << std::endl;
        #endif

        rng.reset();
    }

    // copy memory from one gpu range to another
    void copy(const array_type &from, array_type &to) {
        assert(from.size()==to.size());
        assert(!from.overlaps(to));

        cudaError_t status = cudaMemcpy(
                reinterpret_cast<void*>(to.begin()),
                reinterpret_cast<const void*>(from.begin()),
                from.size()*sizeof(value_type),
                cudaMemcpyDeviceToDevice
        );
    }

    // copy memory from host memory to device
    template <class CoordOther>
    void copy( const ArrayView<value_type, CoordOther> &from,
               array_type &to) {
        static_assert(true, "DeviceCoordinator: unable to copy from other Coordinator");
    }

    template <class Alloc>
    std::pair<SynchEvent, array_type>
    copy(const ArrayView<value_type, HostCoordinator<value_type, Alloc>> &from,
         array_type &to) {
        assert(from.size()==to.size());

        #ifndef NDEBUG
        using oType = ArrayView<value_type, HostCoordinator<value_type, Alloc>>;
        std::cout << "synchronous copy from host to device memory :\n  " 
                  << util::pretty_printer<DeviceCoordinator>::print(*this)
                  << "::copy(\n\t"
                  << util::pretty_printer<oType>::print(from) << ",\n\t"
                  << util::pretty_printer<array_type>::print(to) << ")" << std::endl;
        #endif

        cudaError_t status = cudaMemcpy(
                reinterpret_cast<void*>(to.begin()),
                reinterpret_cast<const void*>(from.begin()),
                from.size()*sizeof(value_type),
                cudaMemcpyHostToDevice
        );

        return std::make_pair(SynchEvent(), to);
    }

    template <size_t alignment>
    std::pair<CudaEvent, array_type>
    copy(const ArrayView<
                value_type,
                HostCoordinator<
                    value_type,
                    PinnedAllocator<
                        value_type,
                        alignment>>> &from,
         array_type &to) {
        assert(from.size()==to.size());

        #ifndef NDEBUG
        using oType = ArrayView< value_type, HostCoordinator< value_type, PinnedAllocator< value_type, alignment>>>;
        std::cout << "asynchronous copy from host to device memory :\n  "
                  << util::pretty_printer<DeviceCoordinator>::print(*this)
                  << "::copy(\n\t"
                  << util::pretty_printer<oType>::print(from) << ",\n\t"
                  << util::pretty_printer<array_type>::print(to) << ")" << std::endl;
        #endif

        cudaError_t status = cudaMemcpy(
                reinterpret_cast<void*>(to.begin()),
                reinterpret_cast<const void*>(from.begin()),
                from.size()*sizeof(value_type),
                cudaMemcpyHostToDevice
        );

        CudaEvent event;
        return std::make_pair(event, to);
    }

    // fill memory
    // todo: use thrust?
    //void fill(array_type &rng, const T& value) {
    //}

    // Generate reference objects for a raw pointer.
    // These helpers allow the ArrayView types to return a reference object
    // that can be used by host code to directly manipulate a memory location
    // on the device.
    reference make_reference(value_type* p) {
        return reference(p);
    }

    const_reference make_reference(value_type const* p) const {
        return const_reference(p);
    }
};

} // namespace memory
////////////////////////////////////////////////////////////////////////////////

