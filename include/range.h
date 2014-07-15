#pragma once

#include <ostream>

#include <cassert>

namespace memory {

class Range {
public:
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    Range()
    : left_(0), right_(0)
    {}

    explicit Range(size_type n)
    : left_(0), right_(n)
    {}

    Range(size_type b, size_type e)
    : left_(b), right_(e)
    {}

    Range(Range const& other) = default;

    size_type size() const {
        return right_ - left_;
    }

    size_type left() const {
        return left_;
    }

    size_type right() const {
        return right_;
    }

    void set(size_type b, size_type e) {
        left_ = b;
        right_ = e;
    }

    Range& operator +=(size_type n) {
        left_ += n;
        right_ += n;

        return (*this);
    }

    Range& operator -=(size_type n) {
        left_ -= n;
        right_   -= n;

        return (*this);
    }

    bool operator == (const Range& other) const {
        return (left_ == other.left_) && (right_ == other.right_);
    }

    bool operator != (const Range& other) const {
        return (left_ != other.left_) || (right_ != other.right_);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Iterator to generate a sequence of integral values
    //
    // Derived from input_iterator because it only makes sense to use as a read
    // only iterator: the iterator does not refer to any external memory,
    // instead returns to state that can only change when iterator is
    // incremented via ++ operator
    ///////////////////////////////////////////////////////////////////////////
    class iterator
    : public std::iterator<std::input_iterator_tag, size_type>
    {
    public:
        /*
        iterator(size_type first, size_type end)
        : range_(first, end)
        , index_(step)
        {
            assert(first<=end);
        }
        */

        iterator(size_type first)
        : index_(first)
        {}

        size_type const& operator*() const {
            return index_;
        }

        size_type const* operator->() const {
            return &index_;
        }

        iterator operator++(int) {
            iterator previous(*this);
            ++(*this);
            return previous;
        }

        const iterator* operator++() {
            ++index_;
            return this;
        }

        bool operator == (const iterator& other) const {
            return index_ == other.index_;
        }

        bool operator != (const iterator& other) const {
            return index_ != other.index_;
        }

    private:
        //Range range_;
        size_type index_;
    };
    ///////////////////////////////////////

    iterator begin() const {
        //return iterator(left_, right_);
        return iterator(left_);
    }

    iterator end() const {
        //return iterator(right_, right_);
        return iterator(right_);
    }

private:
    size_type left_;
    size_type right_;
};

static std::ostream& operator << (std::ostream& os, const Range& rng) {
    os << "[" << rng.left() << ":" << rng.right() << "]";
    return os;
}

} // namespace memory
