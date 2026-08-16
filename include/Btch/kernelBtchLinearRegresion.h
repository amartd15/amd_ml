#pragma once

#include "types.h"

//-------------------------------------------------- GRADENT DESCENT -------------------------------------------//

//Performs the calculation of the gradent in GPU
__global__ void lr_gradientDescent(
    const float* d_X, 
    const float* d_y, 
    float* param, 
    float* grad, 
    float* error,
    int n_points, 
    int n_param
);


__global__ void lr_update_param(
    float* param, 
    float* grad, 
    float alpha, 
    int n_points, 
    int n_param
);


//This function encapsulates the process of launching the kernel of the linear regression.
//Only brings back to host memory the parameters matrix, the rest is kept in device memory
__host__ amd_linear_regression BTCHlinearRregresionKernel(
    BTCH_tensor*        X, 
    BTCH_tensor*        y, 
    tensor*             parameters, 
    tensor*             gradient, 
    tensor*             error,
    
    lr_hiperparameters* hp //Short for hiperparameters
);


//-------------------------------------------------- CHECK TOLERANCES -------------------------------------------//

//Identifies if we had a bouncce back
__host__ bool BTCH_compare_mse(
    float* mse_arr,
    float* mse_arr_aux,

    lr_hiperparameters* hp,
    int num_BTCH
);


//Encapsulates the launch of a kernel that calculates the euclidean norm of an horizontal or vertical vector
__host__ void lr_norm(
    tensor* error, 
    float* mse_aux,

    cudaStream_t error_stream,
    cudaEvent_t  error_event
);


//Performs the euclidean norm of a vactor in GPU
__global__ void lr_kernel_norm(float* data, float* value, int size);


//We check wether we had a bounce back
__host__ void lr_check_bounce(float* mse, float* mse_aux, lr_hiperparameters* param, bool* bounce);


//We check wether we had a bounce back
//__global__ void lr_kernel_check_bounce(float* mse, float* mse_aux, lr_hiperparameters* param, bool* bounce);

__global__ void lr_reset_gradent(float* gradent, int n_param);