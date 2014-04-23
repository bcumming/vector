#include <algorithm>

#include "definitions.h"

#include <cassert>

namespace memory {

template <typename T, int N, int W=1>
class Storage {
public :
    typedef T value_type;
    typedef typename types::size_type size_type;
    typedef typename types::difference_type difference_type;

    // compile time constants
    static const size_type size = N;
    static const size_type width = W;
    static const size_type number_of_values = W*N;

    Storage() {
        fill(value_type(0));
    }

    Storage(value_type const& value) {
        fill(value);
    }

    // () operator for accessing a single element using (i,j) index
    // returns the jth value of the ith variable
    T& operator()(size_type i, size_type j) {
        assert(i<size);
        assert(j<width);
        return data_[W*i,j];
    }

    T const& operator()(size_type i, size_type j) const {
        assert(i<size);
        assert(j<width);
        return data_[W*i,j];
    }


    // [] operator for accessing a single element with a single index directly
    // into underlying storage
    T& operator[](size_type i) {
        assert(i<number_of_values);
        return data_[i];
    }

    T const& operator[](size_type i) const {
        assert(i<number_of_values);
        return data_[i];
    }

    void fill(const T& value) {
        std::fill(data_, data_+number_of_values, value);
    }

private:
    value_type data_[number_of_values];
};

} // namespace memory
