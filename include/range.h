#pragma once

#include <cstddef>
#include <type_traits>

namespace memory {

// forward declarations for helper functions
template <typename T> class range;

// tag for final element in a range
struct end_type {};

// tag for complete range
struct all_type { };

namespace{
    end_type end;
    all_type all;
}

template <typename T>
class range {
public:
    typedef T value_type;
    typedef typename std::size_t size_type;
    typedef typename std::ptrdiff_t difference_type;

    typedef value_type* pointer;
    typedef const pointer const_pointer;
    typedef value_type& reference;
    typedef const reference const_reference;

    range(const_pointer ptr, size_type sz) : pointer_(ptr), size_(sz) {}
    range() : pointer_(nullptr), size_(0) {}

    // generate a reference range using an index pair
    // for example, range(2,5) will return a range of length 3
    // that indexes [2,3,4]
    range<T>
    operator() (const size_type& left, const size_type& right) const {
        #ifdef DEBUG
        assert(left<=right);
        assert(right<=size_);
        #endif
        return range<T>(pointer_+left, right-left);
    }

    // generate a reference range using the end marker
    range<T>
    operator() (const size_type& left, end_type) const {
        #ifdef DEBUG
        assert(left<=size_);
        #endif
        return range<T>(pointer_+left, size_-left);
    }

    // generate a reference range using all
    range<T>
    operator() (all_type) const {
        return range<T>(pointer_, size_);
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

    // -----------------------
    // test whether memory overlaps that referenced by other
    template <typename R>
    // assert that both have same value_type
    // assert that either Range or ReferenceRange
    bool overlaps(const R& other) const {
        return( !((this->begin()>=other.end()) || (other.begin()>=this->end())) );
    }

private:

    pointer pointer_;
    size_type size_;
};

// helpers for identifying ranges
template <typename T>
struct is_range : std::false_type {};

template <typename T>
struct is_range<range<T>> : std::true_type {};

} // namespace memory

