#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

// For dealing fith bias or not
enum bias{

    NO_BIAS,
    YES_BIAS
};

// Used in functions to handle memory, specifies the direcction of the data transfer
enum direction{

    HOST_TO_DEVICE,
    DEVICE_TO_HOST
};

// For handling matrices with device and host memory
struct tensor{

    float*                  data_h;
    float*                  data_d;

    int                     columns;
    int                     rows;
};

struct lr_hiperparameters{

    float*                  alpha; //the current learning rate in this iteration
    int*                    current_iter; //the current iteration

    unsigned int            first_iteration;
    unsigned int            iter;
    unsigned int            n_param;
    unsigned int            n_points;

    float                   tol;
    float                   initial_alpha;
    float                   alpha_reduction;
    float                   alpha_min;

    float                   mse;

    bias                    decision;
};

// All the information necessary for performing a linear regression
struct amd_linear_regression{

    tensor*                 point_matrix;
    tensor*                 result_matrix;

    tensor*                 parameters;
    tensor*                 error;
    tensor*                 gradient;

    lr_hiperparameters*     hiperparameters;
};

void inline CUDA_CHECK(cudaError_t err, std::string msg){
    if(err != cudaSuccess){
        std::cout << "Error-> " << msg << "\nCuda error-> " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
};
