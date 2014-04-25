#include <cstddef>

template <typename T, bool> class Range;

template <typename T>
Range<T,false> make_base_range (typename Range<T,false>::const_pointer ptr,
                                typename Range<T,false>::size_type sz);

template <typename T>
Range<T,true> make_reference_range (typename Range<T,true>::const_pointer ptr,
                                    typename Range<T,true>::size_type sz);

// tag for final element in a range
struct end_type {};

// tag for complete range
struct all_type {};

void foo(end_type) {
    std::cout << "endfoo" << std::endl;
}
void foo(all_type) {
    std::cout << "allfoo" << std::endl;
}

namespace{
    end_type end;
    all_type all;
}

template <typename T, bool REF>
class Range {
public:
    typedef T value_type;
    typedef typename std::size_t size_type;
    typedef typename std::ptrdiff_t difference_type;

    typedef value_type* pointer;
    typedef const pointer const_pointer;
    typedef value_type& reference;
    typedef const reference const_reference;


    typedef Range<T,false> base_range_type;
    typedef Range<T,true>  reference_range_type;

    static const bool is_reference=REF;

    Range(const_pointer ptr, size_type sz) : pointer_(ptr), size_(sz) {}

    // generate a reference range using an index pair
    // for example, range(2,5) will return a range of length 3
    // that indexes [2,3,4]
    reference_range_type
    operator() (const size_type& left, const size_type& right) const {
        #ifdef DEBUG
        assert(left<=right);
        assert(right<=size_);
        #endif
        return make_reference_range<T>(pointer_+left, right-left);
    }

    // generate a reference range using the end marker
    reference_range_type
    operator() (const size_type& left, end_type) const {
        #ifdef DEBUG
        assert(left<=size_);
        #endif
        return make_reference_range<T>(pointer_+left, size_-left);
    }

    // generate a reference range using all
    reference_range_type
    operator() (all_type) const {
        return make_reference_range<T>(pointer_, size_);
    }

    // return direct access to data. This should be provided by specializations
    // for a given architecture, or handled at a higher level
    const_reference operator[](size_type i) const {
        #ifdef DEBUG
        assert(i<size_);
        #endif
        return *(pointer_+i);
    }

    reference operator[](size_type i) {
        #ifdef DEBUG
        assert(i<size_);
        #endif
        return *(pointer_+i);
    }

    // return the pointer
    const_pointer data() const {
        return pointer_;
    }

    pointer data() {
        return pointer_;
    }

    // begin and end iterator pairs
    pointer begin() {
        return pointer_;
    }
    const_pointer begin() const {
        return pointer_;
    }

    pointer end() {
        return pointer_+size_;
    }
    const_pointer end() const {
        return pointer_+size_;
    }

    // -----------------------
    bool is_empty() const {
        return size_>0;
    }

    // -----------------------
    size_type size() const {
        return size_;
    }

private:

    pointer pointer_;
    size_type size_;
};


// helpers for generating base range and reference ranges
template <typename T>
Range<T,false> make_base_range (typename Range<T,false>::const_pointer ptr,
                                typename Range<T,false>::size_type sz)
{
    return Range<T,false>(ptr, sz);
};

template <typename T>
Range<T,true> make_reference_range (typename Range<T,true>::const_pointer ptr,
                                    typename Range<T,true>::size_type sz)
{
    return Range<T,true>(ptr, sz);
}
/*
#include <cstdlib>
#include <vector>
#include <iostream>

#include "range.h"

template<typename R>
void print_range(const R& rng) {
    for(int i=0; i<rng.size(); i++)
        std::cout << rng[i] << " ";
    std::cout << std::endl;
}

template<typename R>
void print_range_stats(const R& rng) {
    std::cout << "range has size " << rng.size() << " and is " << (rng.is_empty() ? "not " : "") << "empty" << std::endl;
}

int main(void) {
    std::vector<double> v(10);
    for(int i=0; i<10; i++)
        v[i] = double(i);

    Range<double,false> r(&(v[0]), 10);
    Range<double,true> r1 = r(4,end);
    Range<double,true> r2 = r1(1,3);

    print_range(r);
    print_range(r1);
    print_range(r2);

    foo(end);
    foo(all);

    return 0;
}
 */
