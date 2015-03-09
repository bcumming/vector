#pragma once

#include <iostream>
#include <string>
#include <type_traits>

#include "definitions.hpp"
#include "Range.hpp"
#include "RangeLimits.hpp"

#ifdef WITH_CYME
#include <cyme/cyme.h>
#endif

////////////////////////////////////////////////////////////////////////////////
namespace memory{

// forward declarations
template<typename T, typename Coord>
class Array;

template <typename R, typename T, typename Coord>
class ArrayViewImpl;

template <typename T, typename Coord>
class ArrayReference;

namespace util {
    template <typename T, typename Coord>
    struct type_printer<ArrayReference<T,Coord>> {
        static std::string print() {
            std::stringstream str;
            #if VERBOSE>1
            str << util::white("ArrayReference") << "<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
            #else
            str << util::white("ArrayReference")
                << "<" << type_printer<Coord>::print() << ">";
            #endif
            return str.str();
        }
    };

    template <typename T, typename Coord>
    struct pretty_printer<ArrayReference<T,Coord>> {
        static std::string print(const ArrayReference<T,Coord>& val) {
            std::stringstream str;
            str << type_printer<ArrayReference<T,Coord>>::print()
                << "(size="     << val.size()
                << ", pointer=" << val.data() << ")";
            return str.str();
        }
    };

    template <typename R, typename T, typename Coord>
    struct type_printer<ArrayViewImpl<R, T,Coord>> {
        static std::string print() {
            std::stringstream str;
            #if VERBOSE>1
            str << util::white("ArrayView") << "<" << type_printer<T>::print()
                << ", " << type_printer<Coord>::print() << ">";
            #else
            str << util::white("ArrayView")
                << "<" << type_printer<Coord>::print() << ">";
            #endif
            return str.str();
        }
    };

    template <typename R, typename T, typename Coord>
    struct pretty_printer<ArrayViewImpl<R, T,Coord>> {
        static std::string print(const ArrayViewImpl<R, T,Coord>& val) {
            std::stringstream str;
            str << type_printer<ArrayViewImpl<R, T,Coord>>::print()
                << "(size="     << val.size()
                << ", pointer=" << val.data() << ")";
            return str.str();
        }
    };
}

namespace impl {

    template <typename T>
    struct is_array;

    // metafunction for indicating whether a type is an ArrayView
    template <typename T>
    struct is_array_view : std::false_type {};

    template <typename R, typename T, typename Coord>
    struct is_array_view<ArrayViewImpl<R, T, Coord> > : std::true_type {};

    template <typename T>
    struct is_array_reference : std::false_type {};

    template <typename T, typename Coord>
    struct is_array_reference<ArrayReference<T, Coord> > : std::true_type {};

    template <typename A>
    struct has_array_view_base : std::false_type {};

    // Helper functions that access the reset() methods in ArrayView.
    // Required to work around a bug in nvcc that makes it awkward to give
    // Coordinator classes friend access to ArrayView types, so that the
    // Coordinator can free and allocate memory. The reset() functions
    // below are friend functions of ArrayView, and are placed in memory::impl::
    // because users should not directly modify pointer or size information in an
    // ArrayView.
    template <typename R, typename T, typename Coord>
    void reset(ArrayViewImpl<R, T, Coord> &v, T* ptr, std::size_t s) {
        v.reset(ptr, s);
    }

    template <typename R, typename T, typename Coord>
    void reset(ArrayViewImpl<R, T, Coord> &v) {
        v.reset();
    }
}

// An ArrayView type refers to a sub-range of an Array. It does not own the
// memory, i.e. it is not responsible for allocation and freeing.
// Currently the ArrayRange type has no way of testing whether the memory to
// which it refers is still valid (i.e. whether or not the original memory has
// been freed)
template <typename R, typename T, typename Coord>
class ArrayViewImpl {
public:
#ifdef WITH_CYME
    using cyme_reference        = cyme::wvec<T,cyme::__GETSIMD__()>;
    using cyme_const_reference  = cyme::rvec<T,cyme::__GETSIMD__()>;
#endif

    using array_reference_type = R;
    using value_type = T;
    using coordinator_type = typename Coord::template rebind<value_type>;

    using value_wrapper = Array<value_type, Coord>;

    using size_type       = types::size_type;
    using difference_type = types::difference_type;

    using pointer         = typename coordinator_type::pointer;
    using const_pointer   = typename coordinator_type::const_pointer;
    using reference       = typename coordinator_type::reference;
    using const_reference = typename coordinator_type::const_reference;

    ////////////////////////////////////////////////////////////////////////////
    // CONSTRUCTORS
    ////////////////////////////////////////////////////////////////////////////
    template <
        typename Other,
        typename = typename std::enable_if< impl::is_array<Other>::value >::type
    >
    explicit ArrayViewImpl(Other&& other)
        : pointer_ (other.data()) , size_(other.size())
    {
#if VERBOSE>1
        std::cout << util::green("ArrayView(&&Other)")
                  << util::pretty_printer<typename std::decay<Other>::type>::print(*this)
                  << std::endl;
#endif
    }

    explicit ArrayViewImpl(pointer ptr, size_type n)
    : pointer_ (ptr)
    , size_(n)
    {
#if VERBOSE>1
        std::cout << util::green("ArrayView(pointer, size_type)")
                  << util::pretty_printer<ArrayViewImpl>::print(*this)
                  << std::endl;
#endif
    }

    explicit ArrayViewImpl() {
        reset();
    };

    ////////////////////////////////////////////////////////////////////////////
    // ACCESSORS
    // overload operator() to provide range based access
    ////////////////////////////////////////////////////////////////////////////

    array_reference_type operator()(size_type const& left, size_type const& right) {
#ifndef NDEBUG
        assert(right<=size_ && left<=right);
#endif
        return array_reference_type(pointer_+left, right-left);
    }

    array_reference_type operator()(size_type const& left, end_type) {
#ifndef NDEBUG
        assert(left<=size_);
#endif
        return array_reference_type(pointer_+left, size_-left);
    }

    array_reference_type operator() (all_type) const {
        return array_reference_type(pointer_, size_);
    }

    // access using a Range
    array_reference_type operator()(Range const& range) {
        size_type left = range.left();
        size_type right = range.right();
#ifndef NDEBUG
        assert(right<=size_ && left<=right);
#endif
        return array_reference_type(pointer_+left, right-left);
    }

    template <
        typename Other,
        typename = typename std::enable_if< impl::is_array<Other>::value >::type
    >
    ArrayViewImpl operator=(Other&& other) {
#if VERBOSE>1
        std::cerr << util::pretty_printer<ArrayViewImpl>::print(*this)
                  << "::" << util::blue("operator=") << "("
                  << util::pretty_printer<ArrayViewImpl>::print(other)
                  << ")" << std::endl;
#endif
        reset(other.data(), other.size());
        return *this;
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
    ~ArrayViewImpl() {}

    // test whether memory overlaps that referenced by other
    bool overlaps(const ArrayViewImpl& other) const {
        return( !((this->begin()>=other.end()) || (other.begin()>=this->end())) );
    }

    memory::Range range() const {
        return memory::Range(0, size());
    }

#ifdef WITH_CYME
    cyme_reference
    wvec(size_type i) {
        return cyme::wvec<value_type, cyme::__GETSIMD__()>
                (&pointer_[i*cyme::stride<T,cyme::AoSoA>::helper_stride()]);
    }

    cyme_const_reference
    rvec(size_type i) const {
        return cyme::rvec<value_type, cyme::__GETSIMD__()>
                (&pointer_[i*cyme::stride<T,cyme::AoSoA>::helper_stride()]);
    }
#endif

protected :
    template <typename RR, typename U, typename C>
    friend void impl::reset(ArrayViewImpl<RR, U, C>&, U*, std::size_t);
    template <typename RR, typename U, typename C>
    friend void impl::reset(ArrayViewImpl<RR, U, C>&);

    void swap(ArrayViewImpl& other) {
        auto ptr = other.data();
        auto sz  = other.size();
        other.reset(pointer_, size_);
        pointer_ = ptr;
        size_    = sz;
    }

    void reset() {
        pointer_ = nullptr;
        size_ = 0;
    }

    void reset(pointer ptr, size_type n) {
        pointer_ = ptr;
        size_ = n;
    }

    // disallow constructors that imply allocation of memory
    ArrayViewImpl(const std::size_t &n) {};

    coordinator_type coordinator_;
    pointer          pointer_;
    size_type        size_;
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename T, typename Coord>
class ArrayReference
    : public ArrayViewImpl<ArrayReference<T, Coord>, T, Coord> {
public:
    using base = ArrayViewImpl<ArrayReference<T, Coord>, T, Coord>;
    using value_type = typename base::value_type;
    using size_type  = typename base::size_type;

    using pointer         = typename base::pointer;
    using const_pointer   = typename base::const_pointer;
    using reference       = typename base::reference;
    using const_reference = typename base::const_reference;

    using base::coordinator_;
    using base::pointer_;
    using base::size_;
    using base::size;

    // Make only one valid constructor, for maintenance reasons.
    // The only place where ArrayReference types are created is in the
    // operator() calls in the ArrayViewImpl, so it is not an issue to have
    // one method for creating References
    explicit ArrayReference(pointer ptr, size_type n)
        : base(ptr, n)
    {
#if VERBOSE>1
        std::cout << util::green("ArrayReference(pointer, size_type)")
                  << util::pretty_printer<base>::print(*this)
                  << std::endl;
#endif
    }

    // the operator=() operators are key: they facilitate copying data from
    // you can make these return an event type, for synchronization
    template <
        typename Other,
        typename = typename std::enable_if< impl::is_array<Other>::value >::type
    >
    ArrayReference operator = (Other&& other) {
#ifndef NDEBUG
        assert(other.size() == this->size());
#endif
#ifdef VERBOSE
        std::cerr << util::type_printer<ArrayReference>::print()
                  << "::" << util::blue("operator=") << "(&&"
                  << util::type_printer<typename std::decay<Other>::type>::print()
                  << ")" << std::endl;
#endif
        base::coordinator_.copy(other, *this);

        return *this;
    }

    ArrayReference& operator = (value_type value) {
#ifdef VERBOSE
        std::cerr << util::pretty_printer<ArrayReference>::print(*this)
                  << "::" << util::blue("operator=") << "(" << value << ")"
                  << std::endl;
#endif
        if(size()>0) {
            base::coordinator_.set(*this, value);
        }

        return *this;
    }

private:
    // default constructor
    // A reference can't be default initialized, because they are designed
    // to be temporary objects that facilitate writing to or reading from
    // memory. Given this, a reference may only be initialized to refer to
    // an existing ArrayView, and it makes no sense to allow one to be default
    // initialized with null data
    ArrayReference() {}

};

template <typename T, typename Coord>
using ArrayView = ArrayViewImpl<ArrayReference<T, Coord>, T, Coord>;

} // namespace memory
////////////////////////////////////////////////////////////////////////////////

