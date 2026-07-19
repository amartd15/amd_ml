#pragma once

#include "types.h"

//-------------------------------------------------- GRADENT DESCENT -------------------------------------------//

//Performs the calculation of the SGD in GPU
__global__ void StocasticGradientDescent(
    const float* d_X, 
    const float* d_y, 
    float* param,
    int n_points, int n_param, float learning_rate
);


//This function encapsulates the process of launching the kernel of the SGD linear regression.
//Only brings back to host memory the parameters matrix, the rest is kept in device memory
__host__ void SGDlinearRregresionKernel(
    tensor* X, tensor* y, tensor* parameters, tensor* error,
    unsigned int n_param, unsigned int n_points, unsigned int n_iter, 
    float learning_rate, float desired_tol
);


//-------------------------------------------------- CHECK TOLERANCES -------------------------------------------//

//We calculate the error with our current model with the dataset
__global__ void SGD_calculateError(
    const float* d_X,
    const float* d_y,
    const float* param,
    float* error,
    int n_points, int n_param
);


//Every ITERATION_CHECK_N iterations we check if the tolerance is met, if we detect a bounce back
//We reduce by LEARNING_RATE_REDUCTION the learning rate, until it reches MINIMUM_LEARNING_RATE
__host__ bool SGD_checkError(int iter, float desired_tol, tensor* mse, float* alpha, tensor* error);


//Encapsulates the launch of a kernel that calculates the euclidean norm of an horizontal or vertical vector
__global__ void SGD_norm(float* data, float* norm, int size);


//Performs the euclidean norm of a vactor in GPU
__host__ float SGD_calculateNorm(tensor* vector);
