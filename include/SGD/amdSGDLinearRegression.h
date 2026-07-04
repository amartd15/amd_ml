#pragma once

#include <random>

#include "types.h"
#include "amdMemoryManagement.h"
#include "kernelSGDLinearRegresion.h"


#ifdef __cplusplus
extern "C" {
#endif

// This function performs a classical stocastic gradent descent on a couple of points and 
// return a context with all variables available
// It has the ability to add a bias (independent term) to the data set if needed
// The context shall be cleared with cleanContext()
__host__ amd_linear_regression SGD_linear_regression(
    float* point_matrix, float* result_matrix,
    unsigned int n_points, unsigned int n_parameters, unsigned int n_iter,
    float desired_tolerance, float initial_seed, float learning_rate,
    bias decision
);

#ifdef __cplusplus
}
#endif
