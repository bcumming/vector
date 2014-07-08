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
    // meta function that returns true if x is a power of two (including 1==2^0)
    template <size_t x>
    struct is_power_of_two : std::integral_constant< bool, !(x&(x-1)) > {};

    // meta function that returns the smallest power of two that is strictly greater than x
    template <size_t x, size_t p=1>
    struct next_power_of_two
        : std::integral_constant< size_t, next_power_of_two<x-(x&p), (p<<1) >::value > {};
    template <size_t p>
    struct next_power_of_two<0,p>
        : std::integral_constant< size_t, p > {};

    // metafunction that returns the smallest power of two that is greater than or equal to x
    template <size_t x>
    struct round_up_power_of_two
        : std::integral_constant< size_t, is_power_of_two<x>::value ? x : next_power_of_two<x>::value >
    {};

    // metafunction that returns the smallest power of two that is greater than
    // or equal to sizeof(T), and greater than or equal to sizeof(void*)
    template <typename T>
    struct minimum_possible_alignment
    {
        static const size_t pot = round_up_power_of_two<sizeof(T)>::value;
        static const size_t value = pot < sizeof(void*) ? sizeof(void*) : pot;
    };

    // function that allocates memory with alignment specified as a template parameter
    template <typename T, size_t alignment=minimum_possible_alignment<T>::value>
    T* aligned_malloc(size_t size) {
        // double check that alignment is a multiple of sizeof(void*), a prerequisite for posix_memalign()
        static_assert( !(alignment%sizeof(void*)),
                "alignment is not a multiple of sizeof(void*)");
        static_assert( is_power_of_two<alignment>::value,
                "alignment is not a power of two");
        void *ptr;
        int result = posix_memalign(&ptr, alignment, size*sizeof(T));
        if(result)
            ptr=nullptr;
        return reinterpret_cast<T*>(ptr);
    }

    template <size_t Alignment>
    class AlignedPolicy {
    public:
        void *allocate_policy(size_t size) {
            return reinterpret_cast<void *>(aligned_malloc<char, Alignment>(size));
        }

        void free_policy(void *ptr) {
            free(ptr);
        }
    };

#ifdef WITH_CUDA
    namespace cuda {
        template <size_t Alignment>
        class pinned_policy {
        public:
            void *allocate_policy(size_t size) {
                // first allocate memory with the desired alignment
                void* ptr = reinterpret_cast<void *>(aligned_malloc<char, Alignment>(size));

                if(ptr == nullptr)
                    return nullptr;

                // now register the memory with CUDA
                cudaError_t status  = cudaHostRegister(ptr, size, cudaHostRegisterPortable);

                // check that there were no CUDA errors
                if(status != cudaSuccess) {
                    #ifndef NDEBUG
                    std::cerr << "ERROR :: pinned_policy :: unable to register host memory with with cudaHostRegister" << std::endl;
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

        class device_policy {
        public:
            void *allocate_policy(size_t size) {
                // first allocate memory with the desired alignment
                void* ptr = nullptr;
                cudaError_t status = cudaMalloc(&ptr, size);

                if(status != cudaSuccess) {
                    #ifndef NDEBUG
                    std::cerr << "ERROR :: unable to allocate memory with cudaMalloc" << std::endl;
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
    template<typename U>
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
            str << "aligned_policy<" << Alignment << ">";
            return str.str();
        }
    };

    #ifdef WITH_CUDA
    template <size_t Alignment>
    struct type_printer<impl::cuda::pinned_policy<Alignment>>{
        static std::string print() {
            std::stringstream str;
            str << "pinned_policy<" << Alignment << ">";
            return str.str();
        }
    };

    template <>
    struct type_printer<impl::cuda::device_policy>{
        static std::string print() {
            return std::string("device_policy");
        }
    };
#endif

    template <typename T, typename Policy>
    struct type_printer<Allocator<T,Policy>>{
        static std::string print() {
            std::stringstream str;
            str << "allocator<" << type_printer<T>::print()
                << ", " << type_printer<Policy>::print() << ">";
            return str.str();
        }
    };
} // namespace util

// helper for generating an aligned allocator
template <class T, size_t alignment=impl::minimum_possible_alignment<T>::value>
using aligned_allocator = Allocator<T, impl::AlignedPolicy<alignment>>;

#ifdef WITH_CUDA
// for pinned allocation we set the default alignment to correspond to the
// alignment of a page (4096 bytes), because by default pinned memory is
// allocated at page boundaries.
template <class T, size_t alignment=4096>
using pinned_allocator = Allocator<T, impl::cuda::pinned_policy<alignment>>;

template <class T, size_t alignment=impl::minimum_possible_alignment<T>::value>
using cuda_allocator = Allocator<T, impl::cuda::device_policy>;
#endif

} // namespace memory
