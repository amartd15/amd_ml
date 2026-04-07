cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.5/bin/nvcc)

project(amd_ml LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)

add_library(amd_ml_lib SHARED
    src/amdMemoryManagement.cu
    src/lr/amdLinearRegression.cu
    src/lr/kernelLinearRegresion.cu
)

set_target_properties(amd_ml_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(amd_ml_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(amd_ml_lib PUBLIC cuda)

#La parte del ejecutable
add_executable(amd_ml src/main.cu)

target_link_libraries(amd_ml PRIVATE amd_ml_lib)

#Union con python
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(amd_ml_py
    bindings/amd_ml_bindings.cpp
)

target_link_libraries(amd_ml_py PRIVATE amd_ml_lib)