/* 
 * File:   range_wrapper.h
 * Author: bcumming
 *
 * Created on May 3, 2014, 5:14 PM
 */

#pragma once

#include <type_traits>

#include "range.h"

namespace memory{
    // forward declarations
    template<typename T, typename C>
    struct range_by_value;

    template<typename T, typename C>
    struct range_by_reference;

    // metafunctions for checking range types
    template <typename T>
    struct is_range_by_value : std::false_type {};

    template <typename T, typename C>
    struct is_range_by_value<range_by_value<T, C> > : std::true_type {};

    template <typename T>
    struct is_range_by_reference : std::false_type{};

    template <typename T, typename C>
    struct is_range_by_reference<range_by_reference<T, C> > : std::true_type {};

    template <typename T>
    struct is_range_wrapper
        : std::conditional< is_range_by_value<T>::value || is_range_by_reference<T>::value,
                                     std::true_type,
                                     std::false_type>::type
    {};

    template <typename T, typename Coord>
    class range_by_value
        : public range<T> {
    public:
        typedef range<T> range_type;
        typedef typename Coord::template rebind<T>::other coordinator_type;
        typedef T value_type;

        typedef typename range_type::size_type size_type;
        typedef typename range_type::difference_type difference_type;

        typedef value_type* pointer;
        typedef const pointer const_pointer;
        typedef value_type& reference;
        typedef value_type const& const_reference;

        // default constructor
        range_by_value() : range_type(nullptr, 0) {}

        // constructor by size
        explicit range_by_value(const size_t &n)
            : range_type(coordinator_type().allocate(n))
        {}

        // constructor by size
        explicit range_by_value(const int &n)
            : range_type(coordinator_type().allocate(n))
        {}

        // construct as a copy of another range
        //template <class RangeType>
        //explicit range_by_value(const typename std::enable_if<is_range_wrapper<RangeType>::value, RangeType>::type& other)
        template <class RangeType, class EI= typename std::enable_if<is_range_wrapper<RangeType>::value, void>::type >
        explicit range_by_value(const RangeType& other)
            : range_type(coordinator_type().allocate(other.size()))
        {
            coordinator_.copy(other.as_range(), *this);
        }

        explicit range_by_value(const range_by_value& other)
            : range_type(coordinator_type().allocate(other.size()))
        {
            coordinator_.copy(other.as_range(), *this);
        }

        range_type& as_range() {
            // return reference to this, because a range wrapper is derived from
            // range_type
            return *this;
        }

        const range_type& as_range() const {
            // return reference to this, because a range wrapper is derived from
            // range_type
            return *this;
        }

        // we want to free the memory in a "by value" range
        ~range_by_value() {
            //coordinator_.free(this->data()); 
        }
        
    private:
        //range_type range_;
        coordinator_type coordinator_;
    };

    template <typename T, typename Coord>
    class range_by_reference
        : public range<T> {
    public:
        typedef range<T> range_type;
        typedef typename Coord::template rebind<T>::other coordinator_type;
        typedef T value_type;

        typedef typename range_type::size_type size_type;
        typedef typename range_type::difference_type difference_type;

        typedef value_type* pointer;
        typedef const pointer const_pointer;
        typedef value_type& reference;
        typedef value_type const& const_reference;

        // construct as a reference to another range
        template <
            class RangeType,
            class EI= typename std::enable_if<is_range_wrapper<RangeType>::value, void>::type
        >
        explicit range_by_reference(const RangeType& other)
            :   range_type(other.data(), other.size())
        {}

        range_type& as_range() {
            // return reference to this, because a range wrapper is derived from range_type
            return *this;
        }

        const range_type& as_range() const {
            // return reference to this, because a range wrapper is derived from range_type
            return *this;
        }

        // TODO : do we have an equality operator == for ranges, to test whether they point
        // to the same range in memory? Is it implemented here, or should it use an equality operator 
        // for ranges?

        // do nothing for destructor: we don't own the memory in range
        ~range_by_reference() {}
        
    private:
        // disallow constructors that imply allocation of memory
        range_by_reference() {};
        range_by_reference(const size_t &n) {};

        coordinator_type coordinator_;
    };

} // namespace memory




