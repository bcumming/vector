/* 
 * File:   range_wrapper.h
 * Author: bcumming
 *
 * Created on May 3, 2014, 5:14 PM
 */

#pragma once

#include <type_traits>

#include "detail/array_base.h"

////////////////////////////////////////////////////////////////////////////////
namespace memory{

// forward declarations
template<typename T, typename Coord>
struct Array;

template<typename T, typename Coord>
struct ArrayRange;

namespace util {
    template <typename T, typename Coord>
    struct type_printer<Array<T,Coord>>{
        static std::string print() {
            std::stringstream str;
            str << "range_by_value<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct pretty_printer<Array<T,Coord>>{
        static std::string print(const Array<T,Coord>& val) {
            std::stringstream str;
            str << type_printer<Array<T,Coord>>::print()
                << "(size="     << val.size()
                << ", pointer=" << val.data() << ")";
            return str.str();
        }
    };
}

// metafunctions for checking range types
template <typename T>
struct is_range_by_value : std::false_type {};

template <typename T, typename Coord>
struct is_range_by_value<Array<T, Coord> > : std::true_type {};

template <typename T>
struct is_range_by_reference : std::false_type{};

template <typename T, typename Coord>
struct is_range_by_reference<ArrayRange<T, Coord> > : std::true_type {};

template <typename T>
struct is_range_wrapper
    : std::conditional< is_range_by_value<T>::value || is_range_by_reference<T>::value,
                                 std::true_type,
                                 std::false_type>::type
{};

// array by value
// this wrapper owns the memory in the array
// and is responsible for allocating and freeing memory
template <typename T, typename Coord>
class Array
    : public ArrayBase<T> {
public:
    typedef T value_type;
    typedef ArrayBase<value_type> base;
    typedef typename Coord::template rebind<value_type>::other coordinator_type;

    typedef ArrayRange<value_type, Coord> reference_wrapper;

    typedef typename base::size_type size_type;
    typedef typename base::difference_type difference_type;

    typedef value_type* pointer;
    typedef const pointer const_pointer;
    typedef value_type& reference;
    typedef value_type const& const_reference;

    // default constructor
    Array() : base(nullptr, 0) {}

    // constructor by size
    explicit Array(const size_t &n)
        : base(coordinator_type().allocate(n))
    {
        #ifndef NDEBUG
        std::cerr << "CONSTRUCTOR " << util::pretty_printer<Array>::print(*this) << std::endl;
        #endif
    }

    // constructor by size
    explicit Array(const int &n)
        : base(coordinator_type().allocate(n))
    {
        #ifndef NDEBUG
        std::cerr << "CONSTRUCTOR " << util::pretty_printer<Array>::print(*this) << std::endl;
        #endif
    }

    // construct as a copy of another range
    template <
        class RangeType,
        typename std::enable_if<is_range_wrapper<RangeType>::value, void>::type = 0
    >
    explicit Array(const RangeType& other)
        : base(coordinator_type().allocate(other.size()))
    {
        #ifndef NDEBUG
        std::cerr << "CONSTRUCTOR " << util::pretty_printer<Array>::print(*this) << std::endl;
        #endif

        coordinator_.copy(other.as_range(), *this);
    }

    explicit Array(const Array& other)
        : base(coordinator_type().allocate(other.size()))
    {
        #ifndef NDEBUG
        std::cerr << "CONSTRUCTOR " << util::pretty_printer<Array>::print(*this) << std::endl;
        #endif

        coordinator_.copy(static_cast<base const&>(other), *this);
    }

    // construct a copy from a range
    // TODO: this could be dangerous, because we assume that the pointer in rng is safe to 
    // copy from using our coordinator
    explicit Array(base const& rng)
        :   base(coordinator_type().allocate(rng.size()))
    {
        coordinator_.copy(rng, *this);
    }

    // have to free the memory in a "by value" range
    ~Array() {
        #ifndef NDEBUG
        std::cerr << "DESCTRUCTOR " << util::pretty_printer<Array>::print(*this) << std::endl;
        #endif

        coordinator_.free(*this);
    }

    reference_wrapper operator()(all_type) {
        return reference_wrapper::make_range_by_reference( base::operator()(all) );
    }

    template <typename Ts, typename Te>
    reference_wrapper operator()(Ts s, Te e) {
        return reference_wrapper::make_range_by_reference( base::operator()(s, e) );
    }

    const coordinator_type& coordinator() const {
        return coordinator_;
    }

private:
    coordinator_type coordinator_;
};

// An ArrayRange type refers to a sub-range of an Array. It does not own the
// memory, i.e. it is not responsible for allocation and freeing.
// Currently the ArrayRange type has no way of testing whether the memory to
// which it refers is still valid (i.e. whether or not the original memory has
// been freed)
template <typename T, typename Coord>
class ArrayRange
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
        typename std::enable_if<is_range_wrapper<RangeType>::value>::type = 0
    >
    explicit ArrayRange(const RangeType& other)
        :   base(other.data(), other.size())
    {}

    // construct as a reference to a range_wrapper
    explicit ArrayRange(value_wrapper const& rng)
        :   base(rng.data(), rng.size())
    {}

    static ArrayRange make_range_by_reference(base const& rng) {
        return ArrayRange(rng);
    }

    // TODO : do we have an equality operator == for ranges, to test whether they point
    // to the same range in memory? Is it implemented here, or should it use an equality operator 
    // for ranges?

    // do nothing for destructor: we don't own the memory in range
    ~ArrayRange() {}

private:
    // construct as a reference to a range
    //      this should come with a warning: no checks can be performed to assert that the range
    //      points to valid memory (passing a GPU range to a host wrapper would have disasterous side effects)
    //explicit range_by_reference(range_type const& rng)
    explicit ArrayRange(base const& rng)
        :   base(rng)
    {}

    // disallow constructors that imply allocation of memory
    ArrayRange() {};
    ArrayRange(const size_t &n) {};

    coordinator_type coordinator_;
};

// metafunction for returning reference range type for an arbitrary range
//  NULL case: fall through to here if invalid range wrapper is used
template <typename T, typename specialize=void>
struct get_reference_range{};

template <typename T>
struct get_reference_range<T, typename std::enable_if< is_range_wrapper<T>::value>::type > {
    typedef typename T::coordinator_type Coord;
    typedef typename T::value_type Value;
    typedef ArrayRange<Value, Coord> type;
};

} // namespace memory
////////////////////////////////////////////////////////////////////////////////

