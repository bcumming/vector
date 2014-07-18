vector
======

Prototype simple vector storage with ranges and support for seamless CPU-GPU interopability.

Starts with concept of an array, which is a wrapper around a pointer and a length:

``` cpp
template <class T, class Coordinator>
class ArrayView {
    ...
    size_t size_;
    T* pointer_;
}
```
which are coupled with a stateless Coordinator type, which performs memory operations like allocation, copy, and so on. This sounds like an allocator, and indeed Coordinators take an allocator type as a template parameter. Coordinators can be specialized with allocators that allocate memory in "incompatible" memory spaces, for example GPU device memory or HDF5 storage.

Coordinators are used to copy memory from one memory space to another, and dispatching execution on the data according to where it is located (launch a CUDA version of a kernel if the data is on a GPU, or a host version if on the host)

The ArrayView type binds the (pointer,size) tupple with a memory space. The Array type derives from this, adding constructor and desctructor that use Coordinator to allocate and free memory.

ArrayView <- Array

The concept of simple ranges are also supported:
```cpp
class Range {
    size_t begin_;
    size_t end_;
}
```
which can be used to get an ArrayView of an Array or ArrayView's sub-range.

todo
====

- range splitting
- implement copying between different coordinator types
- types for coordinating asynchronous dispatch of memory transfers

