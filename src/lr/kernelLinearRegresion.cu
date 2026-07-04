#include "lr/kernelLinearRegresion.h"
#include "amdMemoryManagement.h"

//Performs the calculation of the gradent, as well as clearing the gradient and error matrix
__global__ void lr_gradientDescent(
    const float* d_X, const float* d_y, 
    float* param, float* grad, float* error,
    int n_points, int n_param
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

        for(int j = blockDim.x / 2; j > 0; j >>= 1){
            if(tdx < j){
                buffer[tdx] += buffer[tdx + j];
            }

            __syncthreads();
        }

        if(tdx == 0) {atomicAdd(&grad[k], buffer[0]); }

    }

}

//Update the parameters matrix
__global__ void lr_updateParameters(
    float* parameters, float* gradent, 
    float alpha, int n_points, int n_param
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx < n_param){
        parameters[idx] -= gradent[idx] * alpha / n_points; 
    }
}


//This function encapsulates the process of launching the kernel of the linear regression.
//Only brings back to host memory the parameters matrix
__host__ void linearRregresionKernel(
    tensor* X, tensor* y, tensor* parameters, tensor* gradient, tensor* error,
    int n_param, int n_points, int n_iter, 
    float learning_rate, float desired_tol
){

    dim3 numThreads, numBlocks, numThreadsParameters;

    //After some benchmarks the best result is given with 128 - 256 threads per block
    numThreads.x = 128;
    numThreads.y = 1;
    numThreads.z = 1;

    //We check wether there are more parameters or points in the dataset
    int length = (n_param < n_points) ? n_points : n_param;
    
    numBlocks.x = (int) (length + numThreads.x - 1) / numThreads.x;
    numBlocks.y = 1;
    numBlocks.z = 1;

    numThreadsParameters.x = n_param;
    numThreadsParameters.y = 1;
    numThreadsParameters.z = 1;

    int shared_mem = numThreads.x * sizeof(float);

    int iter = 0;
    tensor* mse = createTensor(0.0f, 1, 1);

    //The main loop of the algorithm
    do{
        cudaMemsetAsync(gradient->data_d, 0, n_param * sizeof(float));

        lr_gradientDescent <<<numBlocks, numThreads, shared_mem>>>(
            X->data_d, y->data_d,
            parameters->data_d, gradient->data_d, error->data_d,
            n_points, n_param
        );

        lr_updateParameters <<<1, numThreadsParameters>>>(
            parameters->data_d, gradient->data_d, 
            learning_rate, n_points, n_param
        );

    }while(
        (++iter < n_iter) &&
        (lr_checkError(iter, desired_tol, mse, &learning_rate, error))    
    );

    cudaDeviceSynchronize();

    //We bring back only the parameters
    copyMemory(parameters, DEVICE_TO_HOST);

    //We free the pointer
    freeTensor(mse);
}


//Every 10 iterations we check if the tolerance is met
__host__ bool lr_checkError(int iter, float desired_tol, tensor* mse, float* alpha, tensor* error){
    
    if(iter % 10 == 0){

        //We use an auxiliary variable to compare with the value of the previous iteration so that
        //we can catch when the gradent "bounces" back

        float mse_aux = lr_calculateNorm(error);

        if (
            (*mse->data_h < mse_aux) &&
            (iter != 10) //For not reducing the learning rate at the beggining
        ){
            *alpha = 0.1 * *alpha;

            if(*alpha <= 1e-10) {return false;}

            std::cout << "Se ha cambiado la tasa de aprendizaje en la " << iter << " iteracion. Alpha = "<< *alpha << std::endl;
        }

        *mse->data_h = mse_aux;

        if(*mse->data_h <= desired_tol){
            std::cout << "Se ha alcanzado la tolerancia esperada a las " << iter << " iteraciones\n" << std::endl;
            return false;
        }

        return true;

    }else{
        return true;
    }
}


//A first approach to calculate the euclidean norm of a vector in CPU
__host__ float lr_calculateNorm(tensor* vector){

    tensor* mse_squared = createTensor(0.0f, 1, 1);

    int size;

    //We calculate wether the vecotr is a row or a column
    if(min(vector->rows, vector->columns) == 1){
        size = max(vector->rows, vector->columns);
    }else{
        std::cout << "Error, trying to calculate the norm of a matrix" << std::endl;
    }

    dim3 numThreads = {128, 1, 1};
    dim3 numBlocks  = {
        (int) (size + numThreads.x -1) / numThreads.x,
        1,
        1
    };

    int sharedMem = numThreads.x * sizeof(float);

    lr_norm<<<numBlocks, numThreads, sharedMem>>>(vector->data_d, mse_squared->data_d, size);
    copyMemory(mse_squared, DEVICE_TO_HOST);

    float value = *mse_squared->data_h;

    freeTensor(mse_squared);

    return sqrt(value);
}

__global__ void lr_norm(float* data, float* value, int size){
    
    extern __shared__ float buffer[];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tdx = threadIdx.x;

        buffer[tdx] = (idx < size) ? data[idx] * data[idx] : 0.0f;

        __syncthreads();

        for(int j = blockDim.x / 2; j > 0; j >>= 1){
            if(tdx < j){
                buffer[tdx] += buffer[tdx + j];
            }

            __syncthreads();
        }

        if(tdx == 0) {atomicAdd(value, buffer[0]);}

}