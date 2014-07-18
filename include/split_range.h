#pragma once

#include <iterator>
#include <ostream>

#include <cassert>

#include "range.h"

namespace memory {

class SplitRange {
  public:
    typedef Range::size_type size_type;
    typedef Range::difference_type difference_type;

    // split range into n chunks
    SplitRange(Range const& rng, size_type n) : range_(rng) {
        // it makes no sense to break a range into 0 chunks
        assert(n>0);

        // add one to step_ if n does not evenly subdivide the target range
        step_ = rng.size()/n + (rng.size()%n ? 1 : 0);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Iterator to generate a sequence of ranges that split the range into
    // n disjoint sets
    //
    // Derived from input_iterator because it only makes sense to use as a read
    // only iterator: the iterator does not refer to any external memory,
    // instead returns to state that can only change when iterator is
    // incremented via ++ operator
    ///////////////////////////////////////////////////////////////////////////
    class iterator
      : public std::iterator<std::input_iterator_tag, Range>
    {
      public:
          iterator(size_type first, size_type end, size_type step)
              : range_(first, first+step),
                step_(step),
                end_(end)
          {
              assert(first<=end);

              if(range_.right()>end)
                  range_.set(first, end);
          }

          Range const& operator*() const {
              return range_;
          }

          Range const* operator->() const {
              return &range_;
          }

          iterator operator++(int) {
              iterator previous(*this);
              ++(*this);
              return previous;
          }

          const iterator* operator++() {
              // shifting the range
              // we can't just use range_+=step_ in case the original range
              // can't be split into equally-sized sub-ranges
              //
              // this is why we don't have reverse or random access, which would
              // require additional state. It might be nice to create such an
              // iterator, if we are using this method to split up work that is
              // to be passed off to a team of worker threads, so that sub-range
              // lookup can be performed in constant time, not linear time as is
              // the case with a forward iterator.
              size_type first = range_.left()+step_;
              if(first>end_)
                  first=end_;

              size_type last = range_.right()+step_;
              if(last>end_)
                  last=end_;

              // update range
              range_.set(first, last);

              return this;
          }

          bool operator == (const iterator& other) const {
              return range_ == other.range_;
          }

          bool operator != (const iterator& other) const {
              return range_ != other.range_;
          }

      private:
        Range range_;
        size_type end_;    // final value for end
        size_type step_;   // step by which range limits get increased
    };
    ///////////////////////////////////////

    iterator begin() const {
        return iterator(range_.left(), range_.right(), step_);
    }

    iterator end() const {
        return iterator(range_.right(), range_.right(), step_);
    }

    size_type step_size() const {
        return step_;
    }

    Range range() const {
        return range_;
    }

  private:
    size_type step_;
    Range range_;
};

// overload output operator for split range
static std::ostream& operator << (std::ostream& os, const SplitRange& split) {
    os << "(" << split.range() << " by " << split.step_size() << ")";
    return os;
}

}