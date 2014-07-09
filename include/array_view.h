/* 
 * File:   range_wrapper.h
 * Author: bcumming
 *
 * Created on May 3, 2014, 5:14 PM
 */

#pragma once

#include <type_traits>

//#include "detail/array_base.h"
#include "array_base.h"

////////////////////////////////////////////////////////////////////////////////
namespace memory{

// forward declarations
template<typename T, typename Coord>
struct Array;

template<typename T, typename Coord>
struct ArrayView;

template <typename T>
struct is_array_by_value : std::false_type {};

template <typename T, typename Coord>
struct is_array_by_value<Array<T, Coord> > : std::true_type {};

template <typename T>
struct is_array_by_reference : std::false_type{};

template <typename T, typename Coord>
struct is_array_by_reference<ArrayView<T, Coord> > : std::true_type {};

template <typename T>
struct is_array
    : std::conditional< is_array_by_value<T>::value || is_array_by_reference<T>::value,
                                 std::true_type,
                                 std::false_type>::type
{};
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
    explicit ArrayView(value_wrapper const& rng)
        :   base(rng.data(), rng.size())
    {}

    static ArrayView make_array_by_reference(base const& rng) {
        return ArrayView(rng);
    }

    // TODO : do we have an equality operator == for ranges, to test whether they point
    // to the same range in memory? Is it implemented here, or should it use an equality operator 
    // for ranges?

    // do nothing for destructor: we don't own the memory in range
    ~ArrayView() {}

private:
    // construct as a reference to an array
    //      this should come with a warning: no checks can be performed to assert that the range
    //      points to valid memory (passing a GPU range to a host wrapper would have disasterous side effects)
    //explicit array_by_reference(range_type const& rng)
    explicit ArrayView(base const& rng)
        :   base(rng)
    {}

    // disallow constructors that imply allocation of memory
    ArrayView() {};
    ArrayView(const size_t &n) {};

    coordinator_type coordinator_;
};

// metafunction for returning reference range type for an arbitrary range
//  NULL case: fall through to here if invalid range wrapper is used
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

