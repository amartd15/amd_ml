#pragma once

#include "types.h"


__global__ void StocasticGradientDescent(
    const float* d_X, 
    const float* d_y, 
    float* param,
    int n_points, int n_param, float learning_rate
);

__global__ void SGD_calculateError(
    const float* d_X,
    const float* d_y,
    const float* param,
    float* error,
    int n_points, int n_param
);

__global__ void SGD_norm(float* data, float* norm, int size);

__host__ void SGDlinearRregresionKernel(
    tensor* X, tensor* y, tensor* parameters, tensor* gradient, tensor* error,
    int n_param, int n_points, int n_iter, 
    float learning_rate, float desired_tol
);

__host__ bool SGD_checkError(int iter, float desired_tol, tensor* mse, float* alpha, tensor* error);

__host__ float SGD_calculateNorm(tensor* vector);
