#include "lr/kernelLinearRegresion.h"
#include "amdMemoryManagement.h"


//-------------------------------------------------- GRADENT DESCENT -------------------------------------------//

//Performs the calculation of the gradent in GPU
__global__ void lr_gradientDescent(
    const float* d_X, 
    const float* d_y, 
    float* param, 
    float* grad, 
    float* error,
    int n_points, 
    int n_param
){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tdx = threadIdx.x;

    extern __shared__ float buffer[];

    //We create the function and calculate the error for each point
    if(idx < n_points){

        float h = 0.0f;

        #pragma unroll
        for(int k = 0; k < n_param; ++k){
            h += d_X[idx * n_param + k] * param[k];
        }

        error[idx] =  h - d_y[idx];

    }

    //We calculate the gradent for each parameter
    #pragma unroll
    for(int k = 0; k < n_param; ++k){
        buffer[tdx] = (idx < n_points) ? 2 * error[idx] * d_X[idx * n_param + k] : 0.0f;

        __syncthreads();

        //An algorithm to add all the elemtns of an array
        #pragma unroll
        for(int j = blockDim.x / 2; j > 0; j >>= 1){
            if(tdx < j){
                buffer[tdx] += buffer[tdx + j];
            }

            __syncthreads();
        }

        //We add all the results from each block
        if(tdx == 0) {atomicAdd(&grad[k], buffer[0]); }

    }
}


//Updates de parameters in a different kernel
__global__ void lr_update_param(
    float* param, 
    float* grad, 
    float alpha, 
    int n_points, 
    int n_param
){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    //Update the parameters
    if(idx < n_param){
        param[idx] -= grad[idx] * alpha / n_points; 
    }
}


//-------------------------------------------------- LAUNCHING KERNEL -------------------------------------------//


//This function encapsulates the process of launching the kernel of the linear regression.
//Only brings back to host memory the parameters matrix, the rest is kept in device memory
__host__ amd_linear_regression linearRregresionKernel(
    tensor*             X, 
    tensor*             y, 
    tensor*             parameters, 
    tensor*             gradient, 
    tensor*             error,
    
    lr_hiperparameters* hp //Short for hiperparameters
){

    //We check wether there are more parameters or points in the dataset
    int length = (hp->n_param < hp->n_points) ? hp->n_points : hp->n_param;

    //We define the variables needed for launching the kernel
    dim3 numThreads, numBlocks, numBlocksParam;

    numThreads      = {64,                                                      1, 1}; //After some benchmarks the best result is given with 64 - 256 threads per block
    numBlocks       = {(int) (length + numThreads.x - 1) / numThreads.x,        1, 1};
    numBlocksParam  = {(int) (hp->n_param + numThreads.x - 1) / numThreads.x,   1, 1};

    //We define the shared memory we will be using
    int shared_mem = numThreads.x * sizeof(float);

    //An iteration counter
    int iter = 0;
    hp->current_iter = &iter;

    //We create the control variables
    float mse = 0.0f; //To store each iteration the mse
    float* smse_aux = createSharedPointer(0.0f); //To calculate each iteration the mse

    //The main loop of the algorithm
    do{

        //Reset the gradent
        cudaMemset(gradient->data_d, 0, hp->n_param * sizeof(float));

        //Calculate the gradent
        lr_gradientDescent<<<numBlocks, numThreads, shared_mem>>>(
            X->data_d, 
            y->data_d,
            parameters->data_d, 
            gradient->data_d, 
            error->data_d,
            hp->n_points, 
            hp->n_param
        );

        //Update the parameters
        lr_update_param<<<numBlocksParam, numThreads>>>(
            parameters->data_d, 
            gradient->data_d, 
            *hp->alpha, 
            hp->n_points, 
            hp->n_param
        );

    }while(
        (++iter < hp->iter)                                                   &&
        (lr_compare_mse(error, &mse, smse_aux, hp))
    );

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
    context.gradient                 = gradient;
    
    context.hiperparameters          = hp;

    return context;
}


//-------------------------------------------------- CHECK TOLERANCES -------------------------------------------//


//Identifies if we had a bounce back
__host__ bool lr_compare_mse(
    tensor* error, 
    float* mse, 
    float* mse_aux, 
    lr_hiperparameters* hp
){
    
    if(*hp->current_iter % hp->first_iteration != 0) { return true; }

    //Calculate the mse of this iteration
    lr_norm(error, mse_aux);

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


//Encapsulates the launch of a kernel that calculates the euclidean norm of an horizontal or vertical vector
__host__ void lr_norm(
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
    
    lr_kernel_norm<<<numBlocks, numThreads, sharedMem>>>(error->data_d, mse_aux, size);

    //We check for silent errors during the kernel launch
    CUDA_CHECK(cudaGetLastError(), "Lunching the kernels of calculations of norms");
}


//Performs the euclidean norm of a vactor in GPU
__global__ void lr_kernel_norm(
    float* data, 
    float* value, 
    int size
){
    
    extern __shared__ float buffer[];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tdx = threadIdx.x;

    if(idx == 0) { *value = 0; } //Reset the value to 0, the operation is so fast no race conditions should be possible

    //We fill the shared memory with each element of the error squared
    buffer[tdx] = (idx < size) ? data[idx] * data[idx] / size : 0.0f;

    __syncthreads();

    //An algorithm to add all the elements of a vector
    #pragma unroll
    for(int j = blockDim.x / 2; j > 0; j >>= 1){
        if(tdx < j){
            buffer[tdx] += buffer[tdx + j];
        }

        __syncthreads();
    }

    //We copy the result of each block into a variable
    if(tdx == 0) { atomicAdd(value, buffer[0]); }
}
