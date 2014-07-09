#pragma once

#include <cstddef>
#include <type_traits>

#include <definitions.h>

////////////////////////////////////////////////////////////////////////////////
namespace memory {

// forward declarations for helper functions
template <typename T> class ArrayBase;

////////////////////////////////////////////////////////////////////////////////
namespace util {
    // helper function for pretty printing a range
    template <typename T>
    struct type_printer<ArrayBase<T>>{
        static std::string print() {
            std::stringstream str;
            str << "ArrayBase<" << type_printer<T>::print() << ">";
            return str.str();
        }
    };

    template <typename T>
    struct pretty_printer<ArrayBase<T>>{
        static std::string print(const ArrayBase<T>& val) {
            std::stringstream str;
            str << type_printer<ArrayBase<T>>::print()
                << "(size="     << val.size()
                << ", pointer=" << val.data() << ")";
            return str.str();
        }
    };
}
////////////////////////////////////////////////////////////////////////////////

// tag for final element in a range
struct end_type {};

// tag for complete range
struct all_type { };

namespace{
    end_type end;
    all_type all;
}

template <typename T>
class ArrayBase {
public:
    typedef T value_type;
    typedef typename std::size_t size_type;
    typedef typename std::ptrdiff_t difference_type;

    typedef value_type* pointer;
    typedef const pointer const_pointer;
    typedef value_type& reference;
    typedef value_type const & const_reference;

    ArrayBase(const_pointer ptr, size_type sz) : pointer_(ptr), size_(sz) {
        #ifndef NDEBUG
        std::cerr << "CONSTRUCTER "
                  << util::pretty_printer<ArrayBase>::print(*this)
                  << std::endl;
        #endif
    }

    ArrayBase() : pointer_(nullptr), size_(0) {}

    // generate a reference range using an index pair
    // for example, range(2,5) will return a range of length 3 that indexes [2,3,4]
    ArrayBase<T>
    operator() (const size_type& left, const size_type& right) const {
        #ifndef NDEBUG
        assert(left<=right);
        assert(right<=size_);
        std::cerr << "operator()(" << left << "," << right << ") "
                  << util::pretty_printer<ArrayBase>::print(*this)
                  << std::endl;
        #endif
        return ArrayBase<T>(pointer_+left, right-left);
    }

    // generate a reference range using the end marker
    ArrayBase<T>
    operator() (const size_type& left, end_type) const {
        #ifndef NDEBUG
        assert(left<=size_);
        std::cerr << "operator()(" << left << ", end) "
                  << util::pretty_printer<ArrayBase>::print(*this)
                  << std::endl;
        #endif
        return ArrayBase<T>(pointer_+left, size_-left);
    }

    // generate a reference range using all
    ArrayBase<T>
    operator() (all_type) const {
        #ifndef NDEBUG
        std::cerr << "operator()(all) "
                  << util::pretty_printer<ArrayBase>::print(*this)
                  << std::endl;
        #endif
        return ArrayBase<T>(pointer_, size_);
    }

    // return direct access to data. This should be provided by specializations
    // for a given architecture, or handled at a higher level
    const_reference operator[](size_type i) const {
        #ifndef NDEBUG
        assert(i<size_);
        #endif
        return *(pointer_+i);
    }

    reference operator[](size_type i) {
        #ifndef NDEBUG
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

// helpers for identifying array bases
template <typename T>
struct is_array_base : std::false_type {};

template <typename T>
struct is_array_base<ArrayBase<T>> : std::true_type {};

} // namespace memory
////////////////////////////////////////////////////////////////////////////////
