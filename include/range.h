#pragma once

#include <ostream>

#include <cassert>

namespace memory {

class Range {
  public:
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    Range()
    : ibegin_(0), iend_(0)
    {}

    explicit Range(size_type n)
    : ibegin_(0), iend_(n)
    {}

    Range(size_type b, size_type e)
    : ibegin_(b), iend_(e)
    {}

    Range(Range const& other) = default;

    size_type size() const {
        return iend_ - ibegin_;
    }

    size_type begin() const {
        return ibegin_;
    }

    size_type end() const {
        return iend_;
    }

    void set(size_type b, size_type e) {
        ibegin_ = b;
        iend_ = e;
    }

    Range& operator +=(size_type n) {
        ibegin_ += n;
        iend_ += n;

        return (*this);
    }

    Range& operator -=(size_type n) {
        ibegin_ -= n;
        iend_   -= n;

        return (*this);
    }

    bool operator == (const Range& other) const {
        return (ibegin_ == other.ibegin_) && (iend_ == other.iend_);
    }

    bool operator != (const Range& other) const {
        return (ibegin_ != other.ibegin_) || (iend_ != other.iend_);
    }

  private:
    size_type ibegin_;
    size_type iend_;
};

static std::ostream& operator << (std::ostream& os, const Range& rng) {
    os << "[" << rng.begin() << ":" << rng.end() << "]";
    return os;
}

} // namespace memory
