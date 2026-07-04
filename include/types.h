#pragma once

#include <cuda_runtime.h>
#include <iostream>

// For handling matrices with device and host memory
struct tensor{
    float* data_h;
    float* data_d;

    int columns;
    int rows;
};

// For dealing fith bias or not
enum bias{
    NO_BIAS,
    YES_BIAS
};

// All the information necessary for performing a linear regression
struct amd_linear_regression{
    tensor* parameters;
    tensor* error;
    tensor* point_matrix;
    tensor* result_matrix;
    tensor* gradient;

    bias decision;
    float mse;
};

// Used in functions to handle memory, specifies the direcction of the data transfer
enum direction{
    HOST_TO_DEVICE,
    DEVICE_TO_HOST
};
