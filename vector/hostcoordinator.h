#pragma once

namespace memory {
template <typename T, int Alignment=1>
class host_coordinator {
    typedef *T pointer;
    typedef const *T const_pointer;
    typedef &T reference;
    typedef const &T const_reference;

    typedef typename types::size_type size_type;
    typedef typename types::difference_type difference_type;

    T* allocate(size_type n) {
    }
}

} //namespace memory
