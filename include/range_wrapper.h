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
template<typename T, typename Coord>
struct range_by_value;

template<typename T, typename Coord>
struct range_by_reference;

namespace util {
    template <typename T, typename Coord>
    struct type_printer<range_by_value<T,Coord>>{
        static std::string print() {
            std::stringstream str;
            str << "range_by_value<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct pretty_printer<range_by_value<T,Coord>>{
        static std::string print(const range_by_value<T,Coord>& val) {
            std::stringstream str;
            str << type_printer<range_by_value<T,Coord>>::print()
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
struct is_range_by_value<range_by_value<T, Coord> > : std::true_type {};

template <typename T>
struct is_range_by_reference : std::false_type{};

template <typename T, typename Coord>
struct is_range_by_reference<range_by_reference<T, Coord> > : std::true_type {};

template <typename T>
struct is_range_wrapper
    : std::conditional< is_range_by_value<T>::value || is_range_by_reference<T>::value,
                                 std::true_type,
                                 std::false_type>::type
{};

// range by value
// this wrapper owns the memory in the range
// and is responisbile for allocating and freeing memory
template <typename T, typename Coord>
class range_by_value
    : public range<T> {
public:
    typedef T value_type;
    typedef range<value_type> range_type;
    typedef typename Coord::template rebind<value_type>::other coordinator_type;

    typedef range_by_reference<value_type, Coord> reference_wrapper;

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
    {
        #ifndef NDEBUG
        std::cerr << "CONSTRUCTOR " << util::pretty_printer<range_by_value>::print(*this) << std::endl;
        #endif
    }

    // constructor by size
    explicit range_by_value(const int &n)
        : range_type(coordinator_type().allocate(n))
    {
        #ifndef NDEBUG
        std::cerr << "CONSTRUCTOR " << util::pretty_printer<range_by_value>::print(*this) << std::endl;
        #endif
    }

    // construct as a copy of another range
    template <
        class RangeType,
        typename std::enable_if<is_range_wrapper<RangeType>::value, void>::type = 0
    >
    explicit range_by_value(const RangeType& other)
        : range_type(coordinator_type().allocate(other.size()))
    {
        #ifndef NDEBUG
        std::cerr << "CONSTRUCTOR " << util::pretty_printer<range_by_value>::print(*this) << std::endl;
        #endif

        coordinator_.copy(other.as_range(), *this);
    }

    explicit range_by_value(const range_by_value& other)
        : range_type(coordinator_type().allocate(other.size()))
    {
        #ifndef NDEBUG
        std::cerr << "CONSTRUCTOR " << util::pretty_printer<range_by_value>::print(*this) << std::endl;
        #endif

        coordinator_.copy(other.as_range(), *this);
    }

    // construct a copy from a range
    // TODO: this could be dangerous, because we assume that the pointer in rng is safe to 
    // copy from using our coordinator
    explicit range_by_value(range_type const& rng)
        :   range_type(coordinator_type().allocate(rng.size()))
    {
        coordinator_.copy(rng, *this);
    }

    // return reference to this, because a range wrapper is derived from
    // range_type
    range_type& as_range() {
        return *this;
    }

    const range_type& as_range() const {
        return *this;
    }

    // have to free the memory in a "by value" range
    ~range_by_value() {
        #ifndef NDEBUG
        std::cerr << "DESCTRUCTOR " << util::pretty_printer<range_by_value>::print(*this) << std::endl;
        #endif

        coordinator_.free(*this);
    }

    reference_wrapper operator()(all_type) {
        return reference_wrapper::make_range_by_reference( range_type::operator()(all) );
    }

    template <typename Ts, typename Te>
    reference_wrapper operator()(Ts s, Te e) {
        return reference_wrapper::make_range_by_reference( range_type::operator()(s, e) );
    }

    const coordinator_type& coordinator() const {
        return coordinator_;
    }

private:
    coordinator_type coordinator_;
};

template <typename T, typename Coord>
class range_by_reference
    : public range<T> {
public:
    typedef range<T> range_type;
    typedef typename Coord::template rebind<T>::other coordinator_type;
    typedef T value_type;

    typedef range_by_value<T, Coord> value_wrapper;

    typedef typename range_type::size_type size_type;
    typedef typename range_type::difference_type difference_type;

    typedef value_type* pointer;
    typedef const pointer const_pointer;
    typedef value_type& reference;
    typedef value_type const& const_reference;

    // construct as a reference to a range_wrapper
    template <
        class RangeType,
        typename std::enable_if<is_range_wrapper<RangeType>::value>::type = 0
    >
    explicit range_by_reference(const RangeType& other)
        :   range_type(other.data(), other.size())
    {}

    // construct as a reference to a range_wrapper
    explicit range_by_reference(value_wrapper const& rng)
        :   range_type(rng.data(), rng.size())
    {}


    static range_by_reference make_range_by_reference(range_type const& rng) {
        return range_by_reference(rng);
    }

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
    // construct as a reference to a range
    //      this should come with a warning: no checks can be performed to assert that the range
    //      points to valid memory (passing a GPU range to a host wrapper would have disasterous side effects)
    //explicit range_by_reference(range_type const& rng)
    explicit range_by_reference(range_type const& rng)
        :   range_type(rng)
    {}

    // disallow constructors that imply allocation of memory
    range_by_reference() {};
    range_by_reference(const size_t &n) {};

    coordinator_type coordinator_;

    //friend class value_wrapper;
};

// metafunction for returning reference range type for an arbitrary range
//  NULL case: fall through to here if invalid range wrapper is used
template <typename T, typename specialize=void>
struct get_reference_range{};

template <typename T>
struct get_reference_range<T, typename std::enable_if< is_range_wrapper<T>::value>::type > {
    typedef typename T::coordinator_type Coord;
    typedef typename T::value_type Value;
    typedef range_by_reference<Value, Coord> type;
};

/*
// helper for wrapping a refernce wrapper around a range
template <class Range, class Coord>
range_by_value<Range, Coord>
make_reference_range(Range const &rng){
    typedef range_by_value<Range, Coord> ref_type;
    return cast<ref_type>(rng);
}
*/

} // namespace memory

