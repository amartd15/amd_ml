#include "SGD/kernelSGDLinearRegresion.h"
#include "SGD/amdSGDLinearRegression.h"

#ifdef __cplusplus
extern "C"{
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
){
    //We prepare the tensor for the points matrix
    tensor* matrix_points = preparePointsTenstor(point_matrix, n_parameters, n_points, decision);

    //We create the auxiliary tensors
    tensor* matrix_result = createTensor(result_matrix, n_points, 1);
    tensor* parameters = createTensor(initial_seed, n_parameters, 1);
    tensor* error = createTensor(n_points, 1);
    tensor* gradient = createTensor(1, 1);

    //We call the real function
    SGDlinearRregresionKernel(
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
    context.mse = SGD_calculateNorm(error);

    //We return all the calculated data in a structure
    return context;
}

#ifdef __cplusplus
}
#endif