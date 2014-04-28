#pragma once

#include <cstddef>

namespace memory {

// forward declarations for helper functions
template <typename T> class Range;
template <typename T> class ReferenceRange;

//template <typename T>
//ReferenceRange make_reference_range (typename Range<T>::const_pointer ptr,
//                                     typename Range<T>::size_type sz);

// tag for final element in a range
struct end_type {};

// tag for complete range
struct all_type {};

namespace{
    end_type end;
    all_type all;
}

template <typename T>
class Range {
public:
    typedef T value_type;
    typedef typename std::size_t size_type;
    typedef typename std::ptrdiff_t difference_type;

    typedef value_type* pointer;
    typedef const pointer const_pointer;
    typedef value_type& reference;
    typedef const reference const_reference;

    Range(const_pointer ptr, size_type sz) : pointer_(ptr), size_(sz) {}

    // generate a reference range using an index pair
    // for example, range(2,5) will return a range of length 3
    // that indexes [2,3,4]
    ReferenceRange<T>
    operator() (const size_type& left, const size_type& right) const {
        #ifdef DEBUG
        assert(left<=right);
        assert(right<=size_);
        #endif
        return ReferenceRange<T>(pointer_+left, right-left);
    }

    // generate a reference range using the end marker
    ReferenceRange<T>
    operator() (const size_type& left, end_type) const {
        #ifdef DEBUG
        assert(left<=size_);
        #endif
        return ReferenceRange<T>(pointer_+left, size_-left);
    }

    // generate a reference range using all
    ReferenceRange<T>
    operator() (all_type) const {
        return ReferenceRange<T>(pointer_, size_);
    }

    // return direct access to data. This should be provided by specializations
    // for a given architecture, or handled at a higher level
    const_reference
    operator[](size_type i) const {
        #ifdef DEBUG
        assert(i<size_);
        #endif
        return *(pointer_+i);
    }

    reference
    operator[](size_type i) {
        #ifdef DEBUG
        assert(i<size_);
        #endif
        return *(pointer_+i);
    }

    // resets to NULL pointer and length 0
    // use when memory has been freed
    void reset() {
        pointer_ = 0;
        size_ = 0;
    }

    // set new pointer and range size
    // might be used for realloc
    void set(const_pointer ptr, size_type sz) {
        pointer_ = ptr;
        size_ = sz;
    }

    // return the pointer
    const_pointer
    data() const {
        return pointer_;
    }

    pointer
    data() {
        return pointer_;
    }

    // begin and end iterator pairs
    pointer
    begin() {
        return pointer_;
    }
    const_pointer
    begin() const {
        return pointer_;
    }

    pointer
    end() {
        return pointer_+size_;
    }
    const_pointer end() const {
        return pointer_+size_;
    }

    // -----------------------
    bool is_empty() const {
        return size_==0;
    }

    // -----------------------
    size_type size() const {
        return size_;
    }

private:

    pointer pointer_;
    size_type size_;
};

template <typename T>
class ReferenceRange : public Range<T> {
public:
    typedef Range<T> base;
    typedef typename base::value_type value_type;

    typedef typename base::pointer pointer;
    typedef typename base::const_pointer const_pointer;
    typedef typename base::reference reference;
    typedef typename base::const_reference const_reference;
    typedef typename base::size_type size_type;

    ReferenceRange(const_pointer ptr, size_type sz) : base(ptr, sz) {};

private:
    // disallow creating a NULL range
    // this might be relaxed
    ReferenceRange();
};


// helpers for generating reference range

/*
template <typename T>
ReferenceRange<T>
make_reference_range( typename Range<>::const_pointer ptr,
                      typename Range<T>::size_type sz)
{
    return ReferenceRange<T>(ptr, sz);
}
*/

} // namespace memory

