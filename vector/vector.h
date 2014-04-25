#pragma once

#include "storage.h"

#include "definitions.h"

namespace memory {

// IDEAS
template <typename S, typename Coord>
class Vector{
public:
    typedef typename storage_type::value_type value_type; // todo : should this be storage_type
    typedef S storage_type;
    typedef typename Coord::rebind<typename storage_type::value_type> coordinator;

    typedef *storage_type pointer;
    typedef const *storage_type const_pointer;
    typedef &storage_type reference;
    typedef const &storage_type const_reference;

    typedef typename types::size_type size_type;
    typedef typename types::difference_type difference_type;

    Vector(int n) {
    }
}
} // namespace memory
