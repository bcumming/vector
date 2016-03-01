set(CUDA_PROPOGATE_HOST_FLAGS OFF)

cmake_minimum_required (VERSION 2.8)
project (vector)
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# KNL
#set( KNL_BACKEND "OFF" CACHE BOOL "Support KNL HBM memory" )
option( KNL_BACKEND "Support KNL HBM memory" off )
if( KNL_BACKEND )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_KNL")
endif()

# KNL
option( TBB_SUPPORT "Support for Intel Thread Building Blocks (TBB)" off )
if( TBB_SUPPORT )
    #set(TBB_ROOT_PATH cache string "" )

    #set(TBB_INCLUDE_PATH cache string "${TBB_ROOT_PATH}/include")
    #include_directories("${TBBROOT}/include")

    #set(TBB_LIBRARY_PATH cache string "${TBB_ROOT_PATH}/lib")
    #link_directories(${TBB_INCLUDE_PATH})

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_TBB")
    set(CMAKE_CXX_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -ltbb -ltbbmalloc" )
endif()

# so that std=c++11 won't cause strange warnings when passed via -Xcompiler

# CUDA
option( CUDA_BACKEND "Compile CUDA version" off )
if( CUDA_BACKEND )
    # so that std=c++11 won't cause strange warnings when passed via -Xcompiler
    set(CUDA_PROPOGATE_HOST_FLAGS OFF)

    # need CUDA 6.5 or greater for C++11 support
    find_package( CUDA 6.5 )

    # set the flags to pass to nvcc
    set(CUDA_NVCC_FLAGS -arch=sm_35; -std=c++11)
    add_definitions(-DWITH_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})

    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} $ENV{CRAY_CUDATOOLKIT_POST_LINK_OPTS} -lcudart")
endif()

add_subdirectory(./src)
add_subdirectory(./tests)

