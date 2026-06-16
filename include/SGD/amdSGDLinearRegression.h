#pragma once

#include <random>

#include "types.h"
#include "amdMemoryManagement.h"
#include "kernelSGDLinearRegresion.h"

//Performs a gradient descent algorithm and returns a context with all the
//variables. The context shall be cleared with cleanContext()

#ifdef __cplusplus
extern "C" {
#endif

__host__ amd_linear_regression SGD_linear_regression(
    float* point_matrix, float* result_matrix,
    int n_points, int n_parameters, int n_iter,
    float desired_tolerance, float initial_seed, float learning_rate,
    bias decision
);

#ifdef __cplusplus
}
#endif
