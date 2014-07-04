vector
======

Prototype vector storage.

Has the concept of a memory range, which is a pointer and a length:

``` cpp
template <class T>
class range {
    T* ptr;
    size_t n;
}
```

These are coupled with a stateless Coordinator type, which performs memory operations like allocation, copy, and so on. This sounds like an allocator, and indeed Coordinators take an allocator type as a template parameter. Coordinators can be specialized with allocators that allocate memory in "incompatible" memory spaces, for example GPU device memory or HDF5 storage.

Coordinators are used to copy memory from one memory space to another, and dispatching execution on the data according to where it is located (launch a CUDA version of a kernel if the data is on a GPU, or a host version if on the host)

