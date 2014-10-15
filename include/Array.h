/* 
 * File:   range_wrapper.h
 * Author: bcumming
 *
 * Created on May 3, 2014, 5:14 PM
 */

#pragma once

#include <type_traits>

#include "definitions.h"
#include "ArrayView.h"

////////////////////////////////////////////////////////////////////////////////
namespace memory{

// forward declarations
template<typename T, typename Coord>
struct Array;

namespace util {
    template <typename T, typename Coord>
    struct type_printer<Array<T,Coord>>{
        static std::string print() {
            std::stringstream str;
            str << "Array<" << type_printer<T>::print()
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

// metafunctions for checking array types
template <typename T>
struct is_array_by_value : std::false_type {};

template <typename T, typename Coord>
struct is_array_by_value<Array<T, Coord> > : std::true_type {};

template <typename T>
struct is_array
    : std::conditional< is_array_by_value<T>::value || is_array_by_reference<T>::value,
                        std::true_type,
                        std::false_type>::type
{};

// array by value
// this wrapper owns the memory in the array
// and is responsible for allocating and freeing memory
template <typename T, typename Coord>
class Array
    : public ArrayView<T, Coord> {
public:
    typedef T value_type;
    typedef ArrayView<value_type, Coord> base;
    typedef ArrayView<value_type, Coord> view_type;

    typedef typename Coord::template rebind<value_type>::other coordinator_type;

    typedef typename base::size_type size_type;
    typedef typename base::difference_type difference_type;

    typedef value_type* pointer;
    typedef const pointer const_pointer;
    typedef value_type& reference;
    typedef value_type const& const_reference;

    // default constructor
    // we have to call constructor in ArrayView: pass base
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
    /*
    template <
        class RangeType,
        typename std::enable_if<is_array<RangeType>::value, void>::type = 0
    >
    explicit Array(const RangeType& other)
        : base(coordinator_type().allocate(other.size()))
    {
        coordinator_.copy(static_cast<view_base const&>(other), *this);
    }
    */

    explicit Array(const view_type& other)
        : base(coordinator_type().allocate(other.size()))
    {
        coordinator_.copy(static_cast<base const&>(other), *this);
    }

    explicit Array(const Array& other)
        : base(coordinator_type().allocate(other.size()))
    {
        coordinator_.copy(static_cast<base const&>(other), *this);
    }

    // have to free the memory in a "by value" range
    ~Array() {
        #ifndef NDEBUG
        std::cerr << "DESCTRUCTOR " << util::pretty_printer<Array>::print(*this) << std::endl;
        #endif
        coordinator_.free(*this);
    }

    // use the accessors provided by ArrayView
    // this enforces the requirement that accessing all or a sub-array of an
    // Array should return a view, not a new array.
    using base::operator();

    const coordinator_type& coordinator() const {
        return coordinator_;
    }

private:
    coordinator_type coordinator_;
};

} // namespace memory
////////////////////////////////////////////////////////////////////////////////

