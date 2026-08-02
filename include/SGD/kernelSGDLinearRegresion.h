#pragma once

#include "types.h"

//-------------------------------------------------- GRADENT DESCENT -------------------------------------------//

//Performs the calculation of the SGD in GPU
__global__ void StocasticGradientDescent(
    const float* d_X, 
    const float* d_y, 
    float* param,
    int n_points, 
    int n_param, 
    float learning_rate
);


//This function encapsulates the process of launching the kernel of the SGD linear regression.
//Only brings back to host memory the parameters matrix, the rest is kept in device memory
__host__ amd_linear_regression SGDlinearRregresionKernel(
    tensor*             X, 
    tensor*             y, 
    tensor*             parameters, 
    tensor*             error,
    
    lr_hiperparameters* hp //Short for hiperparameters
);


//-------------------------------------------------- CHECK TOLERANCES -------------------------------------------//

//We calculate the error with our current model with the dataset
__global__ void SGD_calculateError_kernel(
    const float* d_X,
    const float* d_y,
    const float* param,
    float* error,
    int n_points, 
    int n_param
);


//Every ITERATION_CHECK_N iterations we check if the tolerance is met, if we detect a bounce back
//We reduce by LEARNING_RATE_REDUCTION the learning rate, until it reches MINIMUM_LEARNING_RATE
__host__ bool SGD_compare_mse(
    tensor* d_X,
    tensor* d_y,
    tensor* param,
    tensor* error, 
    float* mse, 
    float* mse_aux, 
    lr_hiperparameters* hp
);


//Encapsulates the launch of a kernel that calculates the euclidean norm of an horizontal or vertical vector
__host__ void SGD_calculateError(    
    tensor* d_X,
    tensor* d_y,
    tensor* param,
    tensor* error,
    
    lr_hiperparameters* hp
);


//Performs the euclidean norm of a vactor in GPU
__host__ void SGD_norm(
    tensor* error, 
    float* mse_aux
);

__global__ void SGD_kernel_norm(
    float* data, 
    float* value, 
    int size
);