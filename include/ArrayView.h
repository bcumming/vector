/* 
 * File:   range_wrapper.h
 * Author: bcumming
 *
 * Created on May 3, 2014, 5:14 PM
 */

#pragma once

#include <type_traits>

//#include "array_base.h"
#include "Range.h"

////////////////////////////////////////////////////////////////////////////////
namespace memory{

// tag for final element in a range
struct end_type {};

// tag for complete range
struct all_type { };

namespace{
    end_type end;
    all_type all;
}

// forward declarations
template<typename T, typename Coord>
struct Array;

template<typename T, typename Coord>
struct ArrayView;

namespace util {
    template <typename T, typename Coord>
    struct type_printer<ArrayView<T,Coord>> {
        static std::string print() {
            std::stringstream str;
            str << "ArrayView<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct pretty_printer<ArrayView<T,Coord>> {
        static std::string print(const ArrayView<T,Coord>& val) {
            std::stringstream str;
            str << type_printer<ArrayView<T,Coord>>::print()
                << "(size="     << val.size()
                << ", pointer=" << val.data() << ")";
            return str.str();
        }
    };
}

template <typename T>
struct is_array;

// metafunction for indicating whether a type is an ArrayView
template <typename T>
struct is_array_by_reference : std::false_type{};

template <typename T, typename Coord>
struct is_array_by_reference<ArrayView<T, Coord> > : std::true_type {};

// An ArrayView type refers to a sub-range of an Array. It does not own the
// memory, i.e. it is not responsible for allocation and freeing.
// Currently the ArrayRange type has no way of testing whether the memory to
// which it refers is still valid (i.e. whether or not the original memory has
// been freed)
template <typename T, typename Coord>
class ArrayView {
    // give Coord friendship so it can access private helpers
    friend Coord;
public:
    typedef typename Coord::template rebind<T>::other coordinator_type;
    typedef T value_type;

    typedef Array<T, Coord> value_wrapper;

    typedef typename std::size_t size_type;
    typedef typename std::ptrdiff_t difference_type;

    typedef typename coordinator_type::pointer         pointer;
    typedef typename coordinator_type::const_pointer   const_pointer;
    typedef typename coordinator_type::reference       reference;
    typedef typename coordinator_type::const_reference const_reference;

    ////////////////////////////////////////////////////////////////////////////
    // CONSTRUCTORS
    ////////////////////////////////////////////////////////////////////////////

    // construct as a reference to a range_wrapper
    template <
        class RangeType,
        typename std::enable_if<is_array<RangeType>::value>::type = 0
    >
    explicit ArrayView(RangeType &other)
    : pointer_(other.data())
    , size_(other.size())
    {
#ifndef NDEBUG
        std::cout << "CONSTRUCTOR "
                  << util::pretty_printer<ArrayView>::print(*this)
                  << "(GenericArrayView)"
                  << std::endl;
#endif
    }

    // need this because it isn't caught by the generic constructor above
    // the default constructor, which isn't templated, would be chosen
    //
    // the other argument is not a const reference because an ArrayView can
    // modify it's contents.
    // to make it possible to do const correctness there would need to be a
    // ConstView type
    explicit ArrayView(value_wrapper &other)
    : pointer_ (other.data())
    , size_(other.size())
    {
#ifndef NDEBUG
        std::cout << "CONSTRUCTOR "
                  << util::pretty_printer<ArrayView>::print(*this)
                  << "(ArrayView)"
                  << std::endl;
#endif
    }

    // need this because it isn't caught by the generic constructor above
    // the default constructor, which isn't templated, would be chosen
    explicit ArrayView(pointer ptr, size_type n)
    : pointer_ (ptr)
    , size_(n)
    {
#ifndef NDEBUG
        std::cout << "CONSTRUCTOR "
                  << util::pretty_printer<ArrayView>::print(*this)
                  << "(pointer, size_type)"
                  << std::endl;
#endif
    }

    ////////////////////////////////////////////////////////////////////////////
    // ACCESSORS
    // overload operator() to provide range based access
    ////////////////////////////////////////////////////////////////////////////

    ArrayView operator()(size_type const& left, size_type const& right) {
#ifndef NDEBUG
        assert(right<=size_ && left<=right);
#endif
        return ArrayView(pointer_+left, right-left);
    }

    ArrayView operator()(size_type const& left, end_type) {
#ifndef NDEBUG
        assert(left<=size_);
#endif
        return ArrayView(pointer_+left, size_-left);
    }

    ArrayView operator() (all_type) const {
        return ArrayView(pointer_, size_);
    }

    // access using a Range
    ArrayView operator()(Range const& range) {
        size_type left = range.left();
        size_type right = range.right();
#ifndef NDEBUG
        assert(right<=size_ && left<=right);
#endif
        return ArrayView(pointer_+left, right-left);
    }

    // access to raw data
    pointer data() {
        return pointer_;
    }

    const_pointer data() const {
        return const_pointer(pointer_);
    }

    size_type size() const {
        return size_;
    }

    bool is_empty() const {
        return size_==0;
    }

    // begin/end iterator pairs
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

    // per element accessors
    // return a reference type provided by Coordinator
    reference operator[] (size_type i) {
        return coordinator_.make_reference(pointer_+i);
    }

    const_reference operator[] (size_type i) const {
        return coordinator_.make_reference(pointer_+i);
    }

    // do nothing for destructor: we don't own the memory in range
    ~ArrayView() {}

    // test whether memory overlaps that referenced by other
    bool overlaps(const ArrayView& other) const {
        return( !((this->begin()>=other.end()) || (other.begin()>=this->end())) );
    }

private:
    void reset() {
        pointer_ = nullptr;
        size_ = 0;
    }

    // disallow constructors that imply allocation of memory
    ArrayView() {};
    ArrayView(const size_t &n) {};

    coordinator_type coordinator_;
    pointer pointer_;
    size_type size_;
};

// metafunction to get view type for an Array/ArrayView
template <typename T, typename specialize=void>
struct get_view{};

template <typename T>
struct get_view<T, typename std::enable_if< is_array<T>::value>::type > {
    typedef typename T::coordinator_type Coord;
    typedef typename T::value_type Value;
    typedef ArrayView<Value, Coord> type;
};

} // namespace memory
////////////////////////////////////////////////////////////////////////////////

