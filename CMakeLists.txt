cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.5/bin/nvcc)

project(amd_ml LANGUAGES CXX CUDA)

#Para poder hacer debugging
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose Debug or Release" FORCE)
endif()

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)

#La parte de los ejecutables

option(LR "Descenso en gradiente" ON)
option(SGD "Descenso de Gradiente Estocástico" ON)

#Para si queremos compliar el código de la regresion lineal
if(LR)
    add_library(amd_ml_lr_lib SHARED
    src/amdMemoryManagement.cu
    src/lr/amdLinearRegression.cu
    src/lr/kernelLinearRegresion.cu
    )

    set_target_properties(amd_ml_lr_lib PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

    target_include_directories(amd_ml_lr_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

    target_link_libraries(amd_ml_lr_lib PUBLIC cuda)

    add_executable(amd_ml_lr src/lr/main_lr.cu)

    target_link_libraries(amd_ml_lr PRIVATE amd_ml_lr_lib)

endif()

if(SGD)
    add_library(amd_ml_SGD_lib SHARED
    src/amdMemoryManagement.cu
    src/SGD/amdSGDLinearRegression.cu
    src/SGD/kernelSGDLinearRegresion.cu
    )

    set_target_properties(amd_ml_SGD_lib PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

    target_include_directories(amd_ml_SGD_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

    target_link_libraries(amd_ml_SGD_lib PUBLIC cuda)

    add_executable(amd_ml_SGD src/SGD/main_SGD.cu)

    target_link_libraries(amd_ml_SGD PRIVATE amd_ml_SGD_lib)

endif()

#Union con python
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(amd_ml_py
    bindings/amd_ml_lr_bindings.cpp
)

if(LR)
    target_link_libraries(amd_ml_py PRIVATE amd_ml_lr_lib)
endif()

if(SGD)
    target_link_libraries(amd_ml_py PRIVATE amd_ml_SGD_lib)
endif()