#pragma once

#include <random>

#include "types.h"
#include "amdMemoryManagement.h"
#include "kernelLinearRegresion.h"


//Performs a gradient descent algorithm and returns a context with all the
//variables. The context shall be cleared with cleanContext()
__host__ amd_linear_regression linear_regression(
    float* point_matrix, float* result_matrix,
    int n_points, int n_parameters, int n_iter,
    float desired_tolerance, float initial_seed, float learning_rate,
    bias decision
);
