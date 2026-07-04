#pragma once

#include "types.h"


__global__ void lr_gradientDescent(
    const float* d_X, const float* d_y, 
    float* param, float* grad, float* error,
    int n_points, int n_param
);

__global__ void lr_updateParameters(
    float* parameters, float* gradent, 
    float alpha, int n_points, int n_param
);

__global__ void lr_norm(float* data, float* norm, int size);

__host__ void linearRregresionKernel(
    tensor* X, tensor* y, tensor* parameters, tensor* gradient, tensor* error,
    int n_param, int n_points, int n_iter, 
    float learning_rate, float desired_tol
);

__host__ bool lr_checkError(int iter, float desired_tol, tensor* mse, float* alpha, tensor* error);

__host__ float lr_calculateNorm(tensor* vector);

