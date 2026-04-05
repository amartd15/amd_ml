cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.5/bin/nvcc)

project(amd_ml LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)

file(GLOB_RECURSE SOURCES src/*.cu)

add_executable(amd_ml ${SOURCES})

target_include_directories(amd_ml PRIVATE include)

target_link_libraries(amd_ml PRIVATE cuda)