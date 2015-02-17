#pragma once

#include <limits>

#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "definitions.h"

////////////////////////////////////////////////////////////////////////////////
namespace memory {
////////////////////////////////////////////////////////////////////////////////
namespace impl {
    using size_type = std::size_t;

    /// true if x is a power of two (including 1==2^0)
    constexpr bool
    is_power_of_two(size_type x) {
        return !(x&(x-1));
    }

    /// returns the smallest power of two that is strictly greater than x
    constexpr size_type
    next_power_of_two(size_type x, size_type p) {
        return x==0 ? p : next_power_of_two(x-(x&p), p<<1);
    }

    /// returns the smallest power of two that is greater than or equal to x
    constexpr size_type
    round_up_power_of_two(size_type x) {
        return is_power_of_two(x) ? x : next_power_of_two(x, 1);
    }

    /// returns the smallest power of two that is greater than
    /// or equal to sizeof(T), and greater than or equal to sizeof(void*)
    template <typename T>
    constexpr size_type
    minimum_possible_alignment() {
        return round_up_power_of_two(sizeof(T)) < sizeof(void*)
                    ?   sizeof(void*)
                    :   round_up_power_of_two(sizeof(T));
    }

    /// calculate the padding that has to be added to an array of length n to
    /// ensure that the length of an array is a multiple of alignment
    /// allignment : in bytes
    /// n          : length of array of items with type T
    /// returns    : items of type T require for alignment
    template<typename T>
    size_type
    get_padding(const size_type alignment, size_type n) {
        // calculate the remaninder in bytes for n items of size sizeof(T)
        auto remainder = (n*sizeof(T)) % alignment;
        // calculate padding in bytes
        return remainder ? (alignment - remainder)/sizeof(T)
                         : 0;

        // this is the c++11 constexpr version, which is more difficult to understand
        // turn this on if we need to use this information at compile time
        //return (n*sizeof(T))%alignment
        //    ? (alignment - ((n*sizeof(T))%alignment)) / sizeof(T)
        //    : 0;
    }

    /// function that allocates memory with alignment specified as a template parameter
    template <typename T, size_type alignment=minimum_possible_alignment<T>()>
    T* aligned_malloc(size_type size) {
        // double check that alignment is a multiple of sizeof(void*),
        // which is a prerequisite for posix_memalign()
        static_assert( !(alignment%sizeof(void*)),
                "alignment is not a multiple of sizeof(void*)");
        static_assert( is_power_of_two(alignment),
                "alignment is not a power of two");
        void *ptr;
        int result = posix_memalign(&ptr, alignment, size*sizeof(T));
        if(result)
            ptr=nullptr;
        return reinterpret_cast<T*>(ptr);
    }

    template <size_type Alignment>
    class AlignedPolicy {
    public:
        void *allocate_policy(size_type size) {
            return reinterpret_cast<void *>(aligned_malloc<char, Alignment>(size));
        }

        void free_policy(void *ptr) {
            free(ptr);
        }
    };

#ifdef WITH_CUDA
    namespace cuda {
        template <size_type Alignment>
        class PinnedPolicy {
        public:
            void *allocate_policy(size_type size) {
                // first allocate memory with the desired alignment
                void* ptr = reinterpret_cast<void *>
                                (aligned_malloc<char, Alignment>(size));

                if(ptr == nullptr)
                    return nullptr;

                // now register the memory with CUDA
                cudaError_t status
                    = cudaHostRegister(ptr, size, cudaHostRegisterPortable);

                // check that there were no CUDA errors
                if(status != cudaSuccess) {
                    #ifndef NDEBUG
                    std::cerr << "ERROR :: Pinned :: unable to register host memory with with cudaHostRegister"
                              << std::endl;
                    #endif
                    // free the memory before returning nullptr
                    free(ptr);
                    return nullptr;
                }

                // return our allocated memory
                return ptr;
            }

            void free_policy(void *ptr) {
                if(ptr == nullptr)
                    return;
                cudaHostUnregister(ptr);
                free(ptr);
            }
        };

        class DevicePolicy {
        public:
            void *allocate_policy(size_type size) {
                // first allocate memory with the desired alignment
                void* ptr = nullptr;
                cudaError_t status = cudaMalloc(&ptr, size);

                if(status != cudaSuccess) {
                    #ifndef NDEBUG
                    std::cerr << "ERROR :: unable to allocate memory with cudaMalloc"
                              << std::endl;
                    #endif
                    return nullptr; // return null on failure
                }

                // return our allocated memory
                return ptr;
            }

            void free_policy(void *ptr) {
                if(ptr)
                    cudaFree(ptr);
            }
        };
    } // namespace cuda
#endif
} // namespace impl
////////////////////////////////////////////////////////////////////////////////

template<typename T, typename Policy >
class Allocator : public Policy {
    using Policy::allocate_policy;
    using Policy::free_policy;
public:
    //    typedefs
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

public:
    //    convert an allocator<T> to allocator<U>
    template<typename U>
    struct rebind {
        typedef Allocator<U, Policy> other;
    };

public:
    inline explicit Allocator() {}
    inline ~Allocator() {}
    inline explicit Allocator(Allocator const&) {}

    //    address
    inline pointer address(reference r) { return &r; }
    inline const_pointer address(const_reference r) { return &r; }

    //    memory allocation
    inline pointer allocate(size_type cnt, typename std::allocator<void>::const_pointer = 0) {
        return reinterpret_cast<T*>(allocate_policy(cnt*sizeof(T)));
    }

    inline void deallocate(pointer p, size_type) {
        if( p!=nullptr ) // only free for non-null pointers
            free_policy(p);
    }

    //    size
    inline size_type max_size() const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    //    construction/destruction
    inline void construct(pointer p, const T& t) {
        new(p) T(t);
    }

    inline void destroy(pointer p) {
        p->~T();
    }

    inline bool operator==(Allocator const&) { return true; }
    inline bool operator!=(Allocator const& a) { return !operator==(a); }
};

// pretty printers
namespace util {
    template <size_t Alignment>
    struct type_printer<impl::AlignedPolicy<Alignment>>{
        static std::string print() {
            std::stringstream str;
            str << "AlignedPolicy<" << Alignment << ">";
            return str.str();
        }
    };

    #ifdef WITH_CUDA
    template <size_t Alignment>
    struct type_printer<impl::cuda::PinnedPolicy<Alignment>>{
        static std::string print() {
            std::stringstream str;
            str << "PinnedPolicy<" << Alignment << ">";
            return str.str();
        }
    };

    template <>
    struct type_printer<impl::cuda::DevicePolicy>{
        static std::string print() {
            return std::string("DevicePolicy");
        }
    };
#endif

    template <typename T, typename Policy>
    struct type_printer<Allocator<T,Policy>>{
        static std::string print() {
            std::stringstream str;
            str << "Allocator<" << type_printer<T>::print()
                << ", " << type_printer<Policy>::print() << ">";
            return str.str();
        }
    };
} // namespace util

// helper for generating an aligned allocator
template <class T, size_t alignment=impl::minimum_possible_alignment<T>()>
using AlignedAllocator = Allocator<T, impl::AlignedPolicy<alignment>>;

#ifdef WITH_CUDA
// for pinned allocation we set the default alignment to correspond to the
// alignment of a page (4096 bytes), because pinned memory is allocated at page
// boundaries.
template <class T, size_t alignment=4096>
using PinnedAllocator = Allocator<T, impl::cuda::PinnedPolicy<alignment>>;

template <class T, size_t alignment=impl::minimum_possible_alignment<T>()>
using CudaAllocator = Allocator<T, impl::cuda::DevicePolicy>;
#endif

} // namespace memory
