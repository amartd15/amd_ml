#include "SGD/kernelSGDLinearRegresion.h"
#include "amdMemoryManagement.h"

//Some hiperparameters
#define ITERATION_CHECK_N 10
#define LEARNING_RATE_REDUCTION 0.1f
#define MINIMUM_LEARNING_RATE 1e-10f


//-------------------------------------------------- GRADENT DESCENT -------------------------------------------//

//Performs the calculation of the SGD in GPU
__global__ void StocasticGradientDescent(
    const float* d_X, 
    const float* d_y, 
    float* param,
    int n_points, 
    int n_param, 
    float learning_rate
){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    //We create the function and calculate the error for each point
    if(idx < n_param){

        #pragma unroll
        for(int row = 0; row < n_points; ++row){
            float h = 0.0f;

            #pragma unroll
            for(int k = 0; k < n_param; ++k){
                h += d_X[row * n_param + k] * param[k];
            }

            __syncthreads();

            //We update the parameters
            param[idx] -= 2 * (h - d_y[row]) * d_X[row * n_param + idx] * learning_rate; 

            //As each iteration is done in the same kernel, we have to syncronize the threads manually
            __syncthreads();

        }
    }

}


//This function encapsulates the process of launching the kernel of the SGD linear regression.
//Only brings back to host memory the parameters matrix, the rest is kept in device memory
__host__ amd_linear_regression SGDlinearRregresionKernel(
    tensor*             X, 
    tensor*             y, 
    tensor*             parameters, 
    tensor*             error,
    
    lr_hiperparameters* hp //Short for hiperparameters
){

    //We check wether there are more parameters or points in the dataset
    int length = (hp->n_param < hp->n_points) ? hp->n_points : hp->n_param;

    //We define the variables needed for launching the kernel
    dim3 numThreads, numBlocks;

    numThreads      = {64,                                                      1, 1}; //After some benchmarks the best result is given with 64 - 256 threads per block
    numBlocks       = {(int) (length + numThreads.x - 1) / numThreads.x,        1, 1};

    //An iteration counter
    int iter = 0;
    hp->current_iter = &iter;

    //We create the control variables
    float mse = 0.0f; //To store each iteration the mse
    float* smse_aux = createSharedPointer(0.0f); //To calculate each iteration the mse

    //The main loop of the algorithm
    do{
            StocasticGradientDescent <<<numBlocks, numThreads>>>(
                X->data_d, 
                y->data_d,
                parameters->data_d,
                hp->n_points, 
                hp->n_param, 
                *hp->alpha
            );

    }while(
        (++iter < hp->iter)                                           &&
        
        (
        SGD_compare_mse(
            X,
            y,
            parameters,
            error,
            &mse, 
            smse_aux,
            hp
        )
        )//Second conditional   
    );//while

    //We check for silent errors
    CUDA_CHECK(cudaGetLastError(), "Launching the kernels of the linear regression");

    //We syncronize de device
    CUDA_CHECK(cudaDeviceSynchronize(), "Device syncronization");  

    //We bring back only the parameters
    copyMemory(parameters, DEVICE_TO_HOST);

    //Free shared pointers
    CUDA_CHECK(cudaFree(smse_aux), "Free smse_aux");

    //We get the mse
    hp->mse                          = mse;

    //We create and fill the structure of the function
    amd_linear_regression context;

    context.point_matrix             = X;
    context.result_matrix            = y;

    context.parameters               = parameters;
    context.error                    = error;
    context.gradient                 = createTensor(1, 1); //Empty
    
    context.hiperparameters          = hp;

    return context;
}


//-------------------------------------------------- CHECK TOLERANCES -------------------------------------------//


//We calculate the error with our current model with the dataset
__global__ void SGD_calculateError_kernel(
    const float* d_X,
    const float* d_y,
    const float* param,
    float* error,
    int n_points, 
    int n_param
){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    //We create the function and calculate the error for each point
    if(idx < n_points){

        float h = 0.0f;

        for(int k = 0; k < n_param; ++k){
            h += d_X[idx * n_param + k] * param[k];
        }

        error[idx] =  h - d_y[idx];

    }    
}


//Every ITERATION_CHECK_N iterations we check if the tolerance is met, if we detect a bounce back
//We reduce by LEARNING_RATE_REDUCTION the learning rate, until it reches MINIMUM_LEARNING_RATE
__host__ bool SGD_compare_mse(
    tensor* d_X,
    tensor* d_y,
    tensor* param,
    tensor* error, 
    float* mse, 
    float* mse_aux, 
    lr_hiperparameters* hp
){
    
    if(*hp->current_iter % hp->first_iteration != 0) { return true; }

    //Calculate the error for this iteration
    SGD_calculateError(
        d_X,
        d_y,
        param,
        error,

        hp
    );

    //Calculate the mse of this iteration
    SGD_norm(error, mse_aux);

    //Bring back the result
    CUDA_CHECK(cudaDeviceSynchronize(), "Syncronizing");
    
    //If we detect a bounce back, we modify the learning rate
    if (
        (*mse < *mse_aux) &&
        (*hp->current_iter != hp->first_iteration) //For not reducing the learning rate at the beggining
    ){
        *hp->alpha *= hp->alpha_reduction; //Reduce the learning rate

        if(*hp->alpha <= hp->alpha_min) {return false;}

        std::cout << "Se ha cambiado la tasa de aprendizaje en la " 
                  << *hp->current_iter << " iteracion. Alpha = "
                  << *hp->alpha << std::endl;
    }

    //We swap the values
    *mse = *mse_aux;

    //We check if the tolerance is met
    if(*mse <= hp->tol){
        std::cout << "Se ha alcanzado la tolerancia esperada a las " 
                  << *hp->current_iter 
                  << " iteraciones\n" 
                  << std::endl;
        return false;
    }

    return true;
}


__host__ void SGD_calculateError(    
    tensor* d_X,
    tensor* d_y,
    tensor* param,
    tensor* error,
    
    lr_hiperparameters* hp
){

    //Some parameters to launch the kernel
    dim3 numThreads      = {128,                                                     1, 1};
    dim3 numBlocks       = {(int) (hp->n_param + numThreads.x - 1) / numThreads.x,   1, 1};

    
    SGD_calculateError_kernel<<<numBlocks, numThreads>>>(
    d_X->data_d,
    d_y->data_d,
    param->data_d,
    error->data_d,
    hp->n_points,
    hp->n_param
    );

    //We check for silent errors during the kernel launch
    CUDA_CHECK(cudaGetLastError(), "Lunching the kernels of calculations of norms");
}


//Encapsulates the launch of a kernel that calculates the euclidean norm of an horizontal or vertical vector
__host__ void SGD_norm(
    tensor* error, 
    float* mse_aux
){
    int size;

    //We calculate wether the vector is a row or a column
    if(min(error->rows, error->columns) == 1){
        size = max(error->rows, error->columns);
    }else{
        std::cout << "Error, trying to calculate the norm of a matrix" << std::endl;
        exit(EXIT_FAILURE);
    }

    //Some parameters to launch the kernel
    dim3 numThreads = {128,                                             1, 1};
    dim3 numBlocks  = {(int) (size + numThreads.x -1) / numThreads.x,   1, 1};

    //Shared memory
    int sharedMem = numThreads.x * sizeof(float);

    //We launch the kernel
    cudaMemsetAsync(mse_aux, 0, sizeof(float));
    
    SGD_kernel_norm<<<numBlocks, numThreads, sharedMem>>>(error->data_d, mse_aux, size);

    //We check for silent errors during the kernel launch
    CUDA_CHECK(cudaGetLastError(), "Lunching the kernels of calculations of norms");
}


//Performs the euclidean norm of a vactor in GPU
__global__ void SGD_kernel_norm(
    float* data, 
    float* value, 
    int size
){
    
    extern __shared__ float buffer[];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tdx = threadIdx.x;

    if(idx == 0) { *value = 0; } //Reset the value to 0, the operation is so fast no race conditions should be possible

    //We fill the shared memory with each element of the error squared
    buffer[tdx] = (idx < size) ? data[idx] * data[idx] : 0.0f;

    __syncthreads();

    //An algorithm to add all the elements of a vector
    for(int j = blockDim.x / 2; j > 0; j >>= 1){
        if(tdx < j){
            buffer[tdx] += buffer[tdx + j];
        }

        __syncthreads();
    }

    //We add the result of each block into a variable
    if(tdx == 0) {atomicAdd(value, buffer[0]);}

}