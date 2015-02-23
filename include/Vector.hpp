#pragma once

#include <iostream>

#include "Array.hpp"
#include "definitions.hpp"
#include "HostCoordinator.hpp"

#ifdef WITH_CUDA
#include "DeviceCoordinator.hpp"
#endif


namespace memory {

// specialization for host vectors
template <typename T>
using HostVector = Array<T, HostCoordinator<T>>;
template <typename T>
using HostView = ArrayView<T, HostCoordinator<T>>;

#ifdef WITH_CUDA
// specialization for pinned vectors. Use a host_coordinator, because memory is
// in the host memory space, and all of the helpers (copy, set, etc) are the
// same with and without page locked memory
template <typename T>
using PinnedVector = Array<T, HostCoordinator<T, PinnedAllocator<T>>>;
template <typename T>
using PinnedView = ArrayView<T, HostCoordinator<T, PinnedAllocator<T>>>;

// specialization for device memory
template <typename T>
using DeviceVector = Array<T, DeviceCoordinator<T, CudaAllocator<T>>>;
template <typename T>
using DeviceView = ArrayView<T, DeviceCoordinator<T, CudaAllocator<T>>>;
#endif

} // namespace memory
