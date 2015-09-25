#pragma once

#include <limits>

#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef WITH_KNL
#include <hbwmalloc.h>
#endif

#include "definitions.hpp"
#include "util.hpp"

namespace memory {

namespace impl {
    using size_type = std::size_t;

    constexpr bool
    is_power_of_two(size_type x) {
        return !(x&(x-1));
    }

    // returns the smallest power of two that is strictly greater than x
    constexpr size_type
    next_power_of_two(size_type x, size_type p) {
        return x==0 ? p : next_power_of_two(x-(x&p), p<<1);
    }

    // returns the smallest power of two that is greater than or equal to x
    constexpr size_type
    round_up_power_of_two(size_type x) {
        return is_power_of_two(x) ? x : next_power_of_two(x, 1);
    }

    // returns the smallest power of two that is greater than
    // or equal to sizeof(T), and greater than or equal to sizeof(void*)
    template <typename T>
    constexpr size_type
    minimum_possible_alignment() {
        return round_up_power_of_two(sizeof(T)) < sizeof(void*)
                    ?   sizeof(void*)
                    :   round_up_power_of_two(sizeof(T));
    }

    // Calculate the padding that has to be added to an array of T of length n
    // so that the size of the array in bytes is a multiple of sizeof(T).
    // The returned value is in terms of T, not bytes.
    template<typename T>
    size_type
    get_padding(const size_type alignment, size_type n) {
        // calculate the remaninder in bytes for n items of size sizeof(T)
        auto remainder = (n*sizeof(T)) % alignment;

        // calculate padding in bytes
        return remainder ? (alignment - remainder)/sizeof(T) : 0;
    }

    // allocate memory with alignment specified as a template parameter
    // returns nullptr on failure
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
        if(result) {
            return nullptr;
        }
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

        static constexpr size_type alignment() { return Alignment; }
    };

#ifdef WITH_KNL
    namespace knl {
        // allocate memory with alignment specified as a template parameter
        // returns nullptr on failure
        template <typename T, size_type alignment=minimum_possible_alignment<T>()>
        T* hbw_malloc(size_type size) {
            // double check that alignment is a multiple of sizeof(void*),
            // which is a prerequisite for posix_memalign()
            static_assert( !(alignment%sizeof(void*)),
                    "alignment is not a multiple of sizeof(void*)");
            static_assert( is_power_of_two(alignment),
                    "alignment is not a power of two");
            void *ptr;
            int result = hbw_posix_memalign(&ptr, alignment, size*sizeof(T));
            if(result) {
                return nullptr;
            }
            return reinterpret_cast<T*>(ptr);
        }

        template <size_type Alignment>
        class HBWPolicy {
        public:
            void *allocate_policy(size_type size) {
                return reinterpret_cast<void *>(hbw_malloc<char, Alignment>(size));
            }

            void free_policy(void *ptr) {
                free(ptr);
            }

            static constexpr size_type alignment() { return Alignment; }
        };
    }
#endif

#ifdef WITH_CUDA
    namespace cuda {
        template <size_type Alignment>
        class PinnedPolicy {
        public:
            void *allocate_policy(size_type size) {
                // first allocate memory with the desired alignment
                void* ptr = reinterpret_cast<void *>
                                (aligned_malloc<char, Alignment>(size));

                if(ptr == nullptr) {
                    return nullptr;
                }

                // register the memory with CUDA
                auto status
                    = cudaHostRegister(ptr, size, cudaHostRegisterPortable);

                if(status != cudaSuccess) {
                    std::cerr << util::red("error") << " memory:: unable to "
                              << "register host memory with with cudaHostRegister"
                              << std::endl;
                    free(ptr);
                    return nullptr;
                }

                return ptr;
            }

            void free_policy(void *ptr) {
                if(ptr == nullptr) {
                    return;
                }
                cudaHostUnregister(ptr);
                free(ptr);
            }

            static constexpr size_type alignment() {
                return Alignment;
            }
        };

        class DevicePolicy {
        public:
            void *allocate_policy(size_type size) {
                void* ptr = nullptr;
                auto status = cudaMalloc(&ptr, size);
                if(status != cudaSuccess) {
                    std::cerr << util::red("error")
                              << " CUDA: unable to allocate "
                              << size << " bytes" << std::endl;;
                    exit(-1);
                }

                return ptr;
            }

            void free_policy(void *ptr) {
                if(ptr) {
                    auto status = cudaFree(ptr);
                    if(status != cudaSuccess) {
                        std::cerr << util::red("error")
                                  << " unable to free CUDA pointer "
                                  << ptr << std::endl;;
                        exit(-1);
                    }
                }
            }

            // memory allocated using cudaMalloc has alignment of 256 bytes
            static constexpr size_type alignment() {
                return 256;
            }
        };
    } // namespace cuda
#endif // #ifdef WITH_CUDA
} // namespace impl

template<typename T, typename Policy >
class Allocator : public Policy {
    using Policy::allocate_policy;
    using Policy::free_policy;
public:
    using Policy::alignment;

    using value_type    = T;
    using pointer       = value_type*;
    using const_pointer = const value_type*;
    using reference     = value_type&;
    using const_reference = const value_type& ;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

public:
    template <typename U>
    using rebind = Allocator<U, Policy>;

public:
    inline explicit Allocator() {}
    inline ~Allocator() {}
    inline explicit Allocator(Allocator const&) {}

    inline pointer address(reference r) {
        return &r;
    }
    inline const_pointer address(const_reference r) {
        return &r;
    }

    inline pointer allocate(size_type cnt, typename std::allocator<void>::const_pointer = 0) {
        return reinterpret_cast<T*>(allocate_policy(cnt*sizeof(T)));
    }

    inline void deallocate(pointer p, size_type) {
        if( p!=nullptr )
            free_policy(p);
    }

    inline size_type max_size() const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    inline void construct(pointer p, const T& t) {
        new(p) T(t);
    }

    inline void destroy(pointer p) {
        p->~T();
    }

    inline bool operator==(Allocator const&) {
        return true;
    }
    inline bool operator!=(Allocator const& a) {
        return !operator==(a);
    }
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
            str << util::white("Allocator") << "<" << type_printer<T>::print()
                << ", " << type_printer<Policy>::print() << ">";
            return str.str();
        }
    };
} // namespace util

// helper for generating an aligned allocator
template <class T, size_t alignment=impl::minimum_possible_alignment<T>()>
using AlignedAllocator = Allocator<T, impl::AlignedPolicy<alignment>>;

#ifdef WITH_KNL
// align with 512 bit vector register size
template <class T, size_t alignment=(512/8)>
using HBWAllocator = Allocator<T, impl::knl::HBWPolicy<alignment>>;
#endif

#ifdef WITH_CUDA
// for pinned allocation set the default alignment to correspond to the
// alignment of a page (4096 bytes), because pinned memory is allocated at page
// boundaries.
template <class T, size_t alignment=4096>
using PinnedAllocator = Allocator<T, impl::cuda::PinnedPolicy<alignment>>;

// use 256 as default allignment, because that is the default for cudaMalloc
template <class T, size_t alignment=256>
using CudaAllocator = Allocator<T, impl::cuda::DevicePolicy>;
#endif

} // namespace memory
