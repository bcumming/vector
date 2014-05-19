#pragma once

#include <limits>

namespace memory {
    namespace impl {
        // meta function that returns true if x is a power of two (including 1)
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

        // metafunction that returns the smallest power of two that is greater than or equal to sizeof(T),
        // and greater than or equal to sizeof(void*)
        template <typename T>
        struct minimum_possible_alignment
        {
            static const size_t pot = round_up_power_of_two<sizeof(T)>::value;
            static const size_t value = pot < sizeof(void*) ? sizeof(void*) : pot;
        };

        // function that allocates memory with alignment specified as a template parameter
        template <typename T, size_t alignment=minimum_possible_alignment<T>::value >
        T* aligned_malloc(size_t size) {
            // double check that alignment is a multiple of sizeof(void*), as this is a prerequisite
            // for posix_memalign()
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
    } // namespace impl

template<typename T, int alignment_=impl::minimum_possible_alignment<T>::value >
class aligned_allocator {
public:
    //    typedefs
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    static const size_t alignment=alignment_;

public:
    //    convert an allocator<T> to allocator<U>
    template<typename U>
    struct rebind {
        typedef aligned_allocator<U> other;
    };

public:
    inline explicit aligned_allocator() {}
    inline ~aligned_allocator() {}
    inline explicit aligned_allocator(aligned_allocator const&) {}
    template<typename U>
    inline explicit aligned_allocator(aligned_allocator<U> const&) {}

    //    address
    inline pointer address(reference r) { return &r; }
    inline const_pointer address(const_reference r) { return &r; }

    //    memory allocation
    inline pointer allocate(size_type cnt, typename std::allocator<void>::const_pointer = 0) {
        //return reinterpret_cast<pointer>(::operator new(cnt * sizeof (T)));
        return impl::aligned_malloc<T, alignment_>(cnt);
    }

    inline void deallocate(pointer p, size_type) {
        if( p!=nullptr ) // only free for non-null pointers
            free(p);
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

    inline bool operator==(aligned_allocator const&) { return true; }
    inline bool operator!=(aligned_allocator const& a) { return !operator==(a); }
};

} // namespace memory
