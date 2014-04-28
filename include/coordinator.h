#pragma once

#include <algorithm>
#include <cassert>

#include "definitions.h"

namespace memory {

template <typename T>
class Coordinator {
public:
    typedef T value_type;
    typedef typename types::size_type size_type;
    typedef typename types::difference_type difference_type;

    typedef *storage_type pointer;
    typedef const *storage_type const_pointer;
    typedef &storage_type reference;
    typedef const &storage_type const_reference;

    Coordinator() : data_(0), size_(0) {};
protected:
private:
    value_type *data_;
    size_type size_;
};

} // namespace memory
