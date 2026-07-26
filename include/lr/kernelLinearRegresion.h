#pragma once

#include "types.h"

//-------------------------------------------------- GRADENT DESCENT -------------------------------------------//

//Performs the calculation of the gradent in GPU
__global__ void lr_gradientDescent(
    const float* d_X, const float* d_y, 
    float* param, float* grad, float* error,
    int n_points, int n_param, float* alpha
);


__global__ void lr_update_param(float* param, float* grad, float *alpha, int n_points, int n_param);


//This function encapsulates the process of launching the kernel of the linear regression.
//Only brings back to host memory the parameters matrix, the rest is kept in device memory
__host__ void linearRregresionKernel(
    tensor* X, tensor* y, tensor* parameters, tensor* gradient, tensor* error,
    unsigned int n_param, unsigned int n_points, unsigned int n_iter, 
    float learning_rate, float desired_tol
);


//-------------------------------------------------- CHECK TOLERANCES -------------------------------------------//

//Identifies if we had a bouncce back
__host__ bool lr_compare_mse(tensor* error, float* mse, float* mse_aux, lr_hiperparameters* param, bool* bounce);


//Encapsulates the launch of a kernel that calculates the euclidean norm of an horizontal or vertical vector
__host__ void lr_norm(tensor* error, float* mse_aux);


//Performs the euclidean norm of a vactor in GPU
__global__ void lr_kernel_norm(float* data, float* value, int size);


//We check wether we had a bounce back
__host__ void lr_check_bounce(float* mse, float* mse_aux, lr_hiperparameters* param, bool* bounce);


//We check wether we had a bounce back
__global__ void lr_kernel_check_bounce(float* mse, float* mse_aux, lr_hiperparameters* param, bool* bounce);

