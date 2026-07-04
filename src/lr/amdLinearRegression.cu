#include "lr/kernelLinearRegresion.h"
#include "lr/amdLinearRegression.h"

#ifdef __cplusplus
extern "C" {
#endif

// This function performs a classical gradent descent on a couple of points and 
// return a context with all variables available
// It has the ability to add a bias (independent term) to the data set if needed
// The context shall be cleared with cleanContext()
__host__ amd_linear_regression linear_regression(
    float* point_matrix, float* result_matrix,
    unsigned int n_points, unsigned int n_parameters, unsigned int n_iter,
    float desired_tolerance, float initial_seed, float learning_rate,
    bias decision
){
    //We prepare the tensor for the points matrix
    tensor* matrix_points = preparePointsTenstor(point_matrix, n_parameters, n_points, decision);

    //We create the auxiliary tensors
    tensor* matrix_result = createTensor(result_matrix, n_points, 1); //Column vector
    tensor* parameters = createTensor(initial_seed, n_parameters, 1); //Column vector
    tensor* error = createTensor(n_points, 1); //Column vector
    tensor* gradient = createTensor(n_parameters, 1); //Column vector

    //We call the real function
    linearRregresionKernel(
        matrix_points, matrix_result, parameters, gradient, error,
        n_parameters, n_points, n_iter, 
        learning_rate, desired_tolerance
    );

    //We create and fill the structure of the function
    amd_linear_regression context;

    context.error = error;
    context.parameters = parameters;
    context.point_matrix = matrix_points;
    context.result_matrix = matrix_result;
    context.gradient = gradient;

    context.decision = decision;
    context.mse = lr_calculateNorm(error);

    //We return all the calculated data in a structure
    return context;
}

#ifdef __cplusplus
}
#endif