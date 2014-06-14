set(DRIVER_SOURCES
    driver.cpp
    host_coordinator_unittest.cpp
    range_unittest.cpp
    storage_unittest.cpp
    range_wrapper_unittest.cpp
    host_vector_unittest.cpp
    allocator_unittest.cpp
    gtest-all.cc
)
set(DRIVER_CUDA_SOURCES
    pinned_vector_unittest.cpp
)

if( CUDA_BACKEND )
    set(DRIVER_SOURCES "${DRIVER_SOURCES} ${DRIVER_CUDA_SOURCES}")
endif()

add_executable(driver.exe ${DRIVER_SOURCES})

