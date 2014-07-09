#pragma once

#include <ostream>

#include <cassert>

class Range {
  public:
    Range()
    : ibegin_(0), iend_(0)
    {}

    explicit Range(size_t n)
    : ibegin_(0), iend_(n)
    {}

    Range(size_t b, size_t e)
    : ibegin_(b), iend_(e)
    {}

    Range(Range const& other) = default;

    size_t size() const {
        return iend_ - ibegin_;
    }

    size_t begin() const {
        return ibegin_;
    }

    size_t end() const {
        return iend_;
    }

    void set(size_t b, size_t e) {
        ibegin_ = b;
        iend_ = e;
    }

    Range& operator +=(size_t n) {
        ibegin_ += n;
        iend_ += n;

        return (*this);
    }

    Range& operator -=(size_t n) {
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
    size_t ibegin_;
    size_t iend_;
};

std::ostream& operator << (std::ostream& os, const Range& rng) {
    os << "[" << rng.begin() << ":" << rng.end() << "]";
    return os;
}

