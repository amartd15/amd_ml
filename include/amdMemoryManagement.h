#pragma once

#include "types.h"

__host__ tensor* createTensor(int rows, int cols);
__host__ tensor* createTensor(float* data, int rows, int cols);
__host__ tensor* createTensor(float seed, int rows, int cols);

//Create a unique pointer to a variable in GPU
template<typename T>
__host__ T* createSharedPointer(T value){

    T* aux = (T*)malloc(sizeof(T));

    cudaError_t err = cudaMallocManaged((void**) &aux, sizeof(T));
    if(err != cudaSuccess){
        std::cout << "Error creando memoria compartida. Codigo de error-> " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    *aux = value;

    return aux;
}

__host__ float* allocatePinnedMemory(size_t size);
__host__ float* allocateDeviceMemory(size_t size);
__host__ void copyMemory(tensor* data, direction direction);

__host__ void freeTensor(tensor* data);
__host__ void freeTensor(tensor* data, bias decision);

__host__ tensor* preparePointsTenstor(float* point_matrix, int n_parameters, int n_points, bias decision);

__host__ void cleanContext(amd_linear_regression context);
__host__ void cleanUpDevice();