#include "lr/kernelLinearRegresion.h"
#include "amdMemoryManagement.h"

//Some hiperparameters
#define ITERATION_CHECK_N 10
#define LEARNING_RATE_REDUCTION 0.1f
#define MINIMUM_LEARNING_RATE 1e-10f


//-------------------------------------------------- GRADENT DESCENT -------------------------------------------//

//Performs the calculation of the gradent in GPU
__global__ void lr_gradientDescent(
    const float* d_X, const float* d_y, 
    float* param, float* grad, float* error,
    int n_points, int n_param, float* alpha
){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tdx = threadIdx.x;

    extern __shared__ float buffer[];

    //We create the function and calculate the error for each point
    if(idx < n_points){

        float h = 0.0f;

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
        for(int j = blockDim.x / 2; j > 0; j >>= 1){
            if(tdx < j){
                buffer[tdx] += buffer[tdx + j];
            }

            __syncthreads();
        }

        //We add all the results from each block
        if(tdx == 0) {atomicAdd(&grad[k], buffer[0]); }

    }

    __syncthreads();

    //Updates the parameters, with alpha/n_points as pseudo learning rate
    if(idx < n_param){
        param[idx] -= grad[idx] * *alpha / n_points; 
    }

}


//This function encapsulates the process of launching the kernel of the linear regression.
//Only brings back to host memory the parameters matrix, the rest is kept in device memory
__host__ void linearRregresionKernel(
    tensor* X, tensor* y, tensor* parameters, tensor* gradient, tensor* error,
    unsigned int n_param, unsigned int n_points, unsigned int n_iter, 
    float learning_rate, float desired_tol
){

    //We check wether there are more parameters or points in the dataset
    int length = (n_param < n_points) ? n_points : n_param;

    //We define the variables needed for launching the kernel
    dim3 numThreads, numBlocks;

    numThreads = {128, 1, 1}; //After some benchmarks the best result is given with 128 - 256 threads per block
    numBlocks = {(int) (length + numThreads.x - 1) / numThreads.x, 1, 1};

    //We define the shared memory we will be using
    int shared_mem = numThreads.x * sizeof(float);

    int iter = 0;

    //We create shared variables
    float aux1 = 0.0f;
    float aux2 = 0.0f;
    float* smse             = createSharedPointer(&aux1);
    float* smse_aux         = createSharedPointer(&aux2);
    float* slearning_rate   = createSharedPointer(&learning_rate);

    //We create the structure with the hiperparameters
    lr_hiperparameters* hiperparameters = (lr_hiperparameters*)malloc(sizeof(lr_hiperparameters));

    hiperparameters->alpha           = slearning_rate;
    hiperparameters->current_iter    = &iter;
    hiperparameters->first_iteration = ITERATION_CHECK_N;
    hiperparameters->iter            = n_iter;
    hiperparameters->tol             = desired_tol;
    hiperparameters->initial_alpha   = learning_rate;
    hiperparameters->alpha_reduction = LEARNING_RATE_REDUCTION;
    hiperparameters->alpha_min       = MINIMUM_LEARNING_RATE;

    //The main loop of the algorithm
    do{
        //Reset the gradent
        cudaMemsetAsync(gradient->data_d, 0, n_param * sizeof(float));

        lr_gradientDescent <<<numBlocks, numThreads, shared_mem>>>(
            X->data_d, y->data_d,
            parameters->data_d, gradient->data_d, error->data_d,
            n_points, n_param, slearning_rate
        );

    }while(
        (++iter < n_iter) &&
        (lr_checkError(error, smse, smse_aux, hiperparameters))    
    );

    //We check for silent errors
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cout << "Error during the kernel launch. Error message -> " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        std::cout << "Error synchronizing the device. Error string -> " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }   

    //We bring back only the parameters
    copyMemory(parameters, DEVICE_TO_HOST);

    cudaFree(smse_aux);
}


//-------------------------------------------------- CHECK TOLERANCES -------------------------------------------//

//Every ITERATION_CHECK_N iterations we check if the tolerance is met, if we detect a bounce back
//We reduce by LEARNING_RATE_REDUCTION the learning rate, until it reches MINIMUM_LEARNING_RATE
__host__ bool lr_checkError(tensor* error, float* mse, float* mse_aux, lr_hiperparameters* hiperparam){
    
    if(*hiperparam->current_iter % ITERATION_CHECK_N == 0){

        lr_calculateNorm(error, mse_aux);

        return lr_compare_mse(mse, mse_aux, hiperparam);

    }else{
        return true;
    }
}


//Encapsulates the launch of a kernel that calculates the euclidean norm of an horizontal or vertical vector
__host__ void lr_calculateNorm(tensor* error, float* mse_aux){
        int size;

        //We calculate wether the vector is a row or a column
        if(min(error->rows, error->columns) == 1){
            size = max(error->rows, error->columns);
        }else{
            std::cout << "Error, trying to calculate the norm of a matrix" << std::endl;
            exit(EXIT_FAILURE);
        }

        //Some parameters to launch the kernel
        dim3 numThreads = {128, 1, 1};
        dim3 numBlocks  = {(int) (size + numThreads.x -1) / numThreads.x, 1, 1};

        //Shared memory
        int sharedMem = numThreads.x * sizeof(float);

        //We launch the kernel
        cudaMemset((void*) mse_aux, 0.0f, sizeof(float));
        lr_norm<<<numBlocks, numThreads, sharedMem>>>(error->data_d, mse_aux, size);

        *mse_aux /= size; //According to the formula

        //We check for silent errors during the kernel launch
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
            std::cout << "Error launching the error kernel. Error string -> " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        err = cudaDeviceSynchronize();
        if(err != cudaSuccess){
            std::cout << "Error synchronizing the device. Error string -> " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
}


//Identifies if we had a bouncce back
__host__ bool lr_compare_mse(float* mse, float* mse_aux, lr_hiperparameters* param){
    //If we detect a bounce back, we modify the learning rate
    if (
        (*mse < *mse_aux) &&
        (*param->current_iter != ITERATION_CHECK_N) //For not reducing the learning rate at the beggining
    ){
        *param->alpha *= LEARNING_RATE_REDUCTION; //Reduce the learning rate

        if(*param->alpha <= MINIMUM_LEARNING_RATE) {return false;}

        std::cout << "Se ha cambiado la tasa de aprendizaje en la " 
                  << *param->current_iter << " iteracion. Alpha = "
                  << *param->alpha << std::endl;
    }

    //We swap the values
    *mse = *mse_aux;

    //We check if the tolerance is met
    if(*mse <= param->tol){
        std::cout << "Se ha alcanzado la tolerancia esperada a las " 
                  << *param->current_iter 
                  << " iteraciones\n" 
                  << std::endl;
        return false;
    }

    return true;
}


//Performs the euclidean norm of a vactor in GPU
__global__ void lr_norm(float* data, float* value, int size){
    
    extern __shared__ float buffer[];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tdx = threadIdx.x;

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
    if(tdx == 0) { atomicAdd(value, buffer[0]); }
}