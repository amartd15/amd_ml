#include "SGD/kernelSGDLinearRegresion.h"
#include "SGD/amdSGDLinearRegression.h"

//Some hiperparameters
#define ITERATION_CHECK_N 100
#define LEARNING_RATE_REDUCTION 0.1f
#define MINIMUM_LEARNING_RATE 1e-4f

#ifdef __cplusplus
extern "C"{
#endif

// This function performs a classical stocastic gradent descent on a couple of points and 
// return a context with all variables available
// It has the ability to add a bias (independent term) to the data set if needed
// The context shall be cleared with cleanContext()
__host__ amd_linear_regression SGD_linear_regression(
    float* point_matrix, 
    float* result_matrix,

    unsigned int n_points, 
    unsigned int n_parameters, 
    unsigned int n_iter,

    float desired_tolerance, 
    float initial_seed, 
    float learning_rate,

    bias decision
){

    //We prepare the tensor for the points matrix
    tensor* matrix_points = preparePointsTenstor(point_matrix, n_parameters, n_points, decision);

    //We create the auxiliary tensors
    tensor* matrix_result = createTensor(result_matrix, n_points, 1);
    tensor* parameters = createTensor(initial_seed, n_parameters, 1);
    tensor* error = createTensor(n_points, 1);

    //We define the hiperparameters
    lr_hiperparameters* hiperparameters = (lr_hiperparameters*)malloc(sizeof(lr_hiperparameters));

    hiperparameters->alpha           = &learning_rate;

    hiperparameters->first_iteration = ITERATION_CHECK_N;
    hiperparameters->iter            = n_iter;
    hiperparameters->n_param         = n_parameters;
    hiperparameters->n_points        = n_points;

    hiperparameters->tol             = desired_tolerance;
    hiperparameters->initial_alpha   = learning_rate;
    hiperparameters->alpha_reduction = LEARNING_RATE_REDUCTION;
    hiperparameters->alpha_min       = MINIMUM_LEARNING_RATE;

    hiperparameters->decision        = decision;

    //We call the real function
    amd_linear_regression context = SGDlinearRregresionKernel(
        matrix_points, 
        matrix_result, 
        parameters, 
        error,
        
        hiperparameters
    );

    //We return all the calculated data in a structure
    return context;
}

#ifdef __cplusplus
}
#endif