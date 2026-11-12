cmake_minimum_required(VERSION 3.24)

# Turing (RTX 2070 Super) and Blackwell (RTX 5070). Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 120)
endif()

project(pathtracer LANGUAGES CXX CUDA)

# Benchmark numbers only mean something in Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(bvhcore STATIC
    src/bvh/bvh_io.cpp
    src/bvh/sah_builder.cpp
    src/mesh/obj_loader.cpp
    src/rays/raygen.cpp
    src/trace/trace_dump.cpp
)
target_include_directories(bvhcore PUBLIC
    src/bvh
    src/cuda
    src/mesh
    src/rays
    src/trace
)

add_executable(objinfo src/tools/objinfo_main.cpp)
target_link_libraries(objinfo PRIVATE bvhcore)

add_executable(bvhbuild src/tools/bvhbuild_main.cpp)
target_link_libraries(bvhbuild PRIVATE bvhcore)

add_executable(raygen src/tools/raygen_main.cpp)
target_link_libraries(raygen PRIVATE bvhcore)

add_executable(tracedump src/tools/tracedump_main.cpp)
target_link_libraries(tracedump PRIVATE bvhcore)

enable_testing()
add_executable(bvh_tests tests/test_bvh.cpp)
target_link_libraries(bvh_tests PRIVATE bvhcore)
add_test(NAME bvh_tests COMMAND bvh_tests ${CMAKE_CURRENT_BINARY_DIR})

add_executable(pathtracer src/tools/pathtracer.cu)
target_include_directories(pathtracer PRIVATE src/cuda)

# -lineinfo maps SASS back to source lines in Nsight Compute without disabling optimizations.
target_compile_options(pathtracer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(bruteforce src/tools/bruteforce.cu)
target_link_libraries(bruteforce PRIVATE bvhcore)
target_compile_options(bruteforce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
