#include "SGD/kernelSGDLinearRegresion.h"
#include "SGD/amdSGDLinearRegression.h"

extern "C"{
__host__ amd_linear_regression SGD_linear_regression(
    float* point_matrix, float* result_matrix,
    int n_points, int n_parameters, int n_iter,
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

    return context;
}
}