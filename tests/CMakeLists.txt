set(DRIVER_SOURCES
    driver.cpp
    host_coordinator_unittest.cpp
    #pinned_coordinator_unittest.cpp
    range_unittest.cpp
    storage_unittest.cpp
    range_wrapper_unittest.cpp
    host_vector_unittest.cpp
    pinned_vector_unittest.cpp
    allocator_unittest.cpp
    gtest-all.cc
)
add_executable(driver.exe ${DRIVER_SOURCES})

