#include "Btch/kernelBtchLinearRegresion.h"
#include "Btch/amdBtchLinearRegression.h"

//Some hiperparameters
#define ITERATION_CHECK_N 100
#define LEARNING_RATE_REDUCTION 0.1f
#define MINIMUM_LEARNING_RATE 1e-4f
#define BTCH 1000

#ifdef __cplusplus
extern "C" {
#endif

// This function performs a batch gradent descent on a couple of points and 
// return a context with all variables available
// It has the ability to add a bias (independent term) to the data set if needed
// The context shall be cleared with cleanContext()

//IMPORTANT: WE ASUME THE BATCH FITS IN GPU AND CPU MEMORY
__host__ amd_linear_regression BTCH_linear_regression(
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

    //First we register the point_matrix with pinned memory

    tensor* X = BTCHpreparePointsTenstor(point_matrix, n_parameters, n_points, decision, BTCH);
    tensor* y = BTCHcreateTensor_y(result_matrix, n_points, BTCH);

    //Now we create the rest of the matrices used on each batch iteration
    tensor* gradient = createTensor(n_parameters, 1);
    tensor* error    = createTensor(BTCH        , 1);
    tensor* param    = createTensor(n_parameters, 1);

    //Now we create the hiperparameters
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
    //Current iteration and mse created inside the kernel

    amd_linear_regression context    = BTCHlinearRregresionKernel(
        X,
        y,
        param,
        gradient,
        error,
        hiperparameters
    );

    return context;

}

#ifdef __cplusplus
}
#endif