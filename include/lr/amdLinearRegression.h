#pragma once

#include <random>

#include "types.h"
#include "amdMemoryManagement.h"
#include "kernelLinearRegresion.h"

#ifdef __cplusplus
extern "C" {
#endif

// This function performs a classical gradent descent on a couple of points and 
// return a context with all variables available
// It has the ability to add a bias (independent term) to the data set if needed
// The context shall be cleared with cleanContext()
__host__ amd_linear_regression linear_regression(
    float* point_matrix, float* result_matrix,
    int n_points, int n_parameters, int n_iter,
    float desired_tolerance, float initial_seed, float learning_rate,
    bias decision
);

#ifdef __cplusplus
}
#endif

