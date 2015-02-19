cmake_minimum_required (VERSION 2.8)
project (vector)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")
include_directories(${CMAKE_SOURCE_DIR}/include)

SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

# CUDA
set( CUDA_BACKEND "OFF" CACHE BOOL "Compile CUDA version" )
if( CUDA_BACKEND )
    find_package( CUDA 5.5 )
    set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" FORCE)
    set(CUDA_NVCC_FLAGS "-arch=sm_35 -std=c++11" CACHE STRING "nvcc flags" FORCE)
    add_definitions(-DWITH_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    SET(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} $ENV{CRAY_CUDATOOLKIT_POST_LINK_OPTS} -lcudart")
endif()

add_subdirectory(./tests)
