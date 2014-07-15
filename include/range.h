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

  private:
    size_type left_;
    size_type right_;
};

static std::ostream& operator << (std::ostream& os, const Range& rng) {
    os << "[" << rng.left() << ":" << rng.right() << "]";
    return os;
}

} // namespace memory
