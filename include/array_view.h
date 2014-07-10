/* 
 * File:   range_wrapper.h
 * Author: bcumming
 *
 * Created on May 3, 2014, 5:14 PM
 */

#pragma once

#include <type_traits>

#include "array_base.h"
#include "range.h"

////////////////////////////////////////////////////////////////////////////////
namespace memory{

// forward declarations
template<typename T, typename Coord>
struct Array;

template<typename T, typename Coord>
struct ArrayView;

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
class ArrayView
    : public ArrayBase<T> {
public:
    typedef ArrayBase<T> base;
    typedef typename Coord::template rebind<T>::other coordinator_type;
    typedef T value_type;

    typedef Array<T, Coord> value_wrapper;

    typedef typename base::size_type size_type;
    typedef typename base::difference_type difference_type;

    typedef value_type* pointer;
    typedef const pointer const_pointer;
    typedef value_type& reference;
    typedef value_type const& const_reference;

    // construct as a reference to a range_wrapper
    template <
        class RangeType,
        typename std::enable_if<is_array<RangeType>::value>::type = 0
    >
    explicit ArrayView(const RangeType& other)
        :   base(other.data(), other.size())
    {}

    // construct as a reference to a range_wrapper
    // need this because it isn't caught by the generic constructor above:
    // the default constructor, which isn't templated, would be chosen
    explicit ArrayView(value_wrapper const& rng)
        :   base(rng.data(), rng.size())
    {}

    // this works both whee T1 and T2 are integral types, and where T2 is end_type
    template<typename T1, typename T2>
    ArrayView operator()(T1 const& b, T2 const& e) {
        return ArrayView(base::operator()(b, e));
    }

    // access using a Range
    ArrayView operator()(Range const& range) {
        return ArrayView(base::operator()(range));
    }

    // do nothing for destructor: we don't own the memory in range
    ~ArrayView() {}

protected:
    // this constructor provides an interface for Array, which is derived from
    // ArrayView, to allocate storage from a raw pointer.
    // this functionality isn't exposed, to avoid a user attempting to initialize
    // an array view with a pointer allocated with an incompatible coordinator
    explicit ArrayView(base const& rng)
        :   base(rng)
    {}

private:
    // disallow constructors that imply allocation of memory
    ArrayView() {};
    ArrayView(const size_t &n) {};

    coordinator_type coordinator_;
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

