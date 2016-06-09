#pragma once

#include <algorithm>
#include <memory>
#include <string>

#include "definitions.hpp"
#include "Array.hpp"
#include "Allocator.hpp"

namespace memory {

// forward declare for type printers
template <typename T, class Allocator>
class HostCoordinator;

#ifdef WITH_CUDA
template <typename T, class Allocator>
class DeviceCoordinator;
#endif

namespace util {
    template <typename T, typename Allocator>
    struct type_printer<HostCoordinator<T,Allocator>>{
        static std::string print() {
            #if VERBOSE>1
            return util::white("HostCoordinator") + "<" + type_printer<T>::print()
                   + ", " + type_printer<Allocator>::print() + ">";
            #else
            return util::white("HostCoordinator") + "<" + type_printer<T>::print() + ">";
            #endif
        }
    };

    template <typename T, typename Allocator>
    struct pretty_printer<HostCoordinator<T,Allocator>>{
        static std::string print(const HostCoordinator<T,Allocator>& val) {
            return type_printer<HostCoordinator<T,Allocator>>::print();;
        }
    };
} // namespace util

template <typename T, class Allocator=AlignedAllocator<T> >
class HostCoordinator {
public:
    using value_type = T;

    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using reference       = value_type&;
    using const_reference = value_type const&;
    using view_type       = ArrayView<value_type, HostCoordinator>;
    using size_type       = types::size_type;
    using difference_type = types::difference_type;

    // rebind host_coordinator with another type
    template <typename Tother>
    using rebind = HostCoordinator<Tother, Allocator>;

    view_type allocate(size_type n) {
        typename Allocator::template rebind<value_type> allocator;

        pointer ptr = n>0 ? allocator.allocate(n) : nullptr;

        #ifdef VERBOSE
        std::cerr << util::type_printer<HostCoordinator>::print()
                  << "::" + util::blue("alocate") << "(" << n
                  << " [" << n*sizeof(value_type) << " bytes]) @ " << ptr
                  << std::endl;
        #endif

        return view_type(ptr, n);
    }

    void free(view_type& rng) {
        typename Allocator::template rebind<value_type> allocator;

        if(rng.data()) {
        #ifdef VERBOSE
            std::cerr << util::type_printer<HostCoordinator>::print()
                      << "::" + util::blue("free") << "(" << rng.size()
                      << " [" << rng.size()*sizeof(value_type) << " bytes])"
                      << " @ " << rng.data()
                      << std::endl;

        #endif

            allocator.deallocate(rng.data(), rng.size());
        }

        impl::reset(rng);
    }

    // copy memory between host memory ranges
    template <typename Allocator1, typename Allocator2>
    // requires Allocator1 = Allocator
    // requires Allocator2 = Allocator
    void copy(const ArrayView<value_type, HostCoordinator<value_type, Allocator1>>& from,
                    ArrayView<value_type, HostCoordinator<value_type, Allocator2>>& to)
    {
        assert(from.size()==to.size());
        assert(!from.overlaps(to));

        #ifdef VERBOSE
        using c1 = HostCoordinator<value_type, Allocator1>;
        std::cerr << util::type_printer<c1>::print()
                  << "::" + util::blue("copy") << "(" << from.size()
                  << " [" << from.size()*sizeof(value_type) << " bytes])"
                  << " " << from.data() << util::yellow(" -> ") << to.data()
                  << std::endl;
        #endif

        std::copy(from.begin(), from.end(), to.begin());
    }

    // copy memory between host memory ranges
    template <typename Allocator1, typename Allocator2>
    // requires Allocator1 = Allocator
    // requires Allocator2 = Allocator
    void copy(const ConstArrayView<value_type, HostCoordinator<value_type, Allocator1>>& from,
                    ArrayView<value_type, HostCoordinator<value_type, Allocator2>>& to)
    {
        assert(from.size()==to.size());
        assert(!from.overlaps(to));

        #ifdef VERBOSE
        using c1 = HostCoordinator<value_type, Allocator1>;
        std::cerr << util::type_printer<c1>::print()
                  << "::" + util::blue("copy") << "(" << from.size()
                  << " [" << from.size()*sizeof(value_type) << " bytes])"
                  << " " << from.data() << util::yellow(" -> ") << to.data()
                  << std::endl;
        #endif

        std::copy(from.begin(), from.end(), to.begin());
    }

    /*
    void copy(const view_type &from, view_type &to) {
        assert(from.size()==to.size());
        assert(!from.overlaps(to));

        #ifdef VERBOSE
        std::cerr << util::type_printer<HostCoordinator>::print()
                  << "::" + util::blue("copy") << "(" << from.size()
                  << " [" << from.size()*sizeof(value_type) << " bytes])"
                  << " " << from.data() << util::yellow(" -> ") << to.data()
                  << std::endl;
        #endif

        std::copy(from.begin(), from.end(), to.begin());
    }
    */

#ifdef WITH_CUDA
    // copy memory from device to host
    template <class Alloc>
    void copy(const ArrayView<value_type, DeviceCoordinator<value_type, Alloc>> &from,
              view_type &to) {
        assert(from.size()==to.size());

        #ifdef VERBOSE
        std::cerr << util::type_printer<HostCoordinator>::print()
                  << "::" + util::blue("copy") << "(device2host, " << from.size()
                  << " [" << from.size()*sizeof(value_type) << " bytes])"
                  << " " << from.data() << util::yellow(" -> ") << to.data()
                  << std::endl;
        #endif

        auto status = cudaMemcpy(
                reinterpret_cast<void*>(to.begin()),
                reinterpret_cast<const void*>(from.begin()),
                from.size()*sizeof(value_type),
                cudaMemcpyDeviceToHost
        );
        if(status != cudaSuccess) {
            std::cerr << util::red("error") << " bad CUDA memcopy, unable to copy " << sizeof(T)*from.size() << " bytes from host to device";
            exit(-1);
        }
    }
#endif

    // set all values in a range to val
    void set(view_type &rng, value_type val) {
        #ifdef VERBOSE
        std::cerr << util::type_printer<HostCoordinator>::print()
                  << "::" + util::blue("fill")
                  << "(" << rng.size()  << " * " << val << ")"
                  << " @ " << rng.data()
                  << std::endl;
        #endif
        std::fill(rng.begin(), rng.end(), val);
    }

    reference make_reference(value_type* p) {
        return *p;
    }

    const_reference make_reference(value_type const* p) const {
        return *p;
    }

    static constexpr auto
    alignment() -> decltype(Allocator::alignment()) {
        return Allocator::alignment();
    }

    static constexpr bool
    is_malloc_compatible() {
        return Allocator::is_malloc_compatible();
    }
};

} //namespace memory
