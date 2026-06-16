#pragma once

#include "types.h"


__global__ void StocasticGradientDescent(
    const float* d_X, const float* d_y, 
    float* param, float* grad,
    int n_points, int n_param, int iter, int n_iter
);

__host__ void SGDlinearRregresionKernel(
    tensor* X, tensor* y, tensor* parameters, tensor* gradient, tensor* error,
    int n_param, int n_points, int n_iter, 
    float learning_rate, float desired_tol, float mse
);

__host__ bool checkError(int iter, float desired_tol, float* mse, float* learning_rate, tensor* error);

__host__ float calculateNorm(tensor* vector);