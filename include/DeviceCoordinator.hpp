#pragma once

#include <algorithm>
#include <cstdint>

#include "Allocator.hpp"
#include "Array.hpp"
#include "definitions.hpp"
#include "Event.hpp"
#include "CudaEvent.hpp"
#include "gpu.hpp"

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
            #if VERBOSE > 1
            str << util::white("DeviceCoordinator") << "<"
                << type_printer<T>::print()
                << ", " << type_printer<Allocator>::print() << ">";
            #else
            str << util::white("DeviceCoordinator")
                << "<" << type_printer<T>::print() << ">";
            #endif
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

namespace gpu {
    // brief:
    // We have to perform some type punning to pass arbitrary POD types to the
    // GPU backend without polluting the library front end with CUDA kernels
    // that would require compilation with nvcc.
    //
    // detail:
    // The implementation takes advantage of 4 fill functions that fill GPU
    // memory with a {8, 16, 32, 64} bit unsigned integer. We want to use these
    // functions to fill a block of GPU memory with _any_ 8, 16, 32 or 64 bit POD
    // value. The technique to do this with a 64-bit double, is to first convert
    // the double into a 64-bit unsigned integer (with the same bits, not the
    // same value), then call the 64-bit fill kernel precompiled using nvcc in
    // the gpu library. This technique of converting from one type to another
    // is called type-punning. There are plenty of subtle problems with this, due
    // to C++'s strict aliasing rules, that require memcpy of single bytes if
    // alignment of the two types does not match.

    #define FILL(N) \
    template <typename T> \
    typename std::enable_if<sizeof(T)==sizeof(uint ## N ## _t)>::type \
    fill(T* ptr, T value, size_t n) { \
        using I = uint ## N ## _t; \
        I v; \
        if(alignof(T)==alignof(I)) { \
            *reinterpret_cast<T*>(&v) = value; \
        } \
        else { \
            std::copy_n( \
                reinterpret_cast<char*>(&value), \
                sizeof(T), \
                reinterpret_cast<char*>(&v) \
            ); \
        } \
        fill ## N(reinterpret_cast<I*>(ptr), v, n); \
    }

    FILL(8)
    FILL(16)
    FILL(32)
    FILL(64)
}

template <typename T>
class ConstDeviceReference {
public:
    using value_type = T;
    using pointer = value_type*;

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
    using pointer = value_type*;

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

    using pointer       = value_type*;
    using const_pointer = const value_type*;
    using reference       = DeviceReference<value_type>;
    using const_reference = ConstDeviceReference<value_type>;

    using array_type = ArrayView<value_type, DeviceCoordinator>;

    using size_type       = types::size_type;
    using difference_type = types::difference_type;

    // metafunction for rebinding host_coordinator with another type
    template <typename Tother>
    using rebind = DeviceCoordinator<Tother, Allocator>;

    // allocate memory on the device
    array_type allocate(size_type n) {
        Allocator allocator;

        // only allocate memory if nonzero memory allocation has been requested
        pointer ptr = n>0 ? allocator.allocate(n) : nullptr;

        #ifdef VERBOSE
        std::cerr << util::type_printer<DeviceCoordinator>::print()
                  << util::blue("::allocate") << "(" << n << ")"
                  << (ptr==nullptr && n>0 ? " failure" : " success")
                  << std::endl;
        #endif

        return array_type(ptr, n);
    }

    // free memory on the device
    void free(array_type& rng) {
        Allocator allocator;

        if(rng.data())
            allocator.deallocate(rng.data(), rng.size());

        #ifdef VERBOSE
        std::cerr << util::type_printer<DeviceCoordinator>::print()
                  << "::free()" << std::endl;
        #endif

        impl::reset(rng);
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

        #ifdef VERBOSE
        using oType = ArrayView<value_type, HostCoordinator<value_type, Alloc>>;
        std::cout << util::pretty_printer<DeviceCoordinator>::print(*this)
                  << "::" << util::blue("copy") << "(asynchronous, " << from.size() << ")"
                  << "\n  " << util::type_printer<oType>::print() << " @ " << from.data()
                  << util::yellow(" -> ")
                  << util::type_printer<array_type>::print() << " @ " << to.data() << std::endl;
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

        #ifdef VERBOSE
        using oType = ArrayView< value_type, HostCoordinator< value_type, PinnedAllocator< value_type, alignment>>>;
        std::cout << util::pretty_printer<DeviceCoordinator>::print(*this)
                  << "::" << util::blue("copy") << "(asynchronous, " << from.size() << ")"
                  << "\n  " << util::type_printer<oType>::print() << " @ " << from.data()
                  << util::yellow(" -> ")
                  << util::type_printer<array_type>::print() << " @ " << to.data() << std::endl;
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
    void set(array_type &rng, value_type value) {
        gpu::fill<value_type>(rng.data(), value, rng.size());
    }

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

    static constexpr auto
    alignment() -> decltype(Allocator_::alignment()) {
        return Allocator_::alignment();
    }
};

} // namespace memory
////////////////////////////////////////////////////////////////////////////////

