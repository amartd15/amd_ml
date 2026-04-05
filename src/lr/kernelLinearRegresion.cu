#include "lr/kernelLinearRegresion.h"
#include "amdMemoryManagement.h"

//Performs the calculation of the gradent, as well as clearing the gradient and error matrix
__global__ void gradientDescent(
    const float* d_X, const float* d_y, 
    float* param, float* grad, float* error,
    int n_points, int n_param
){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tdx = threadIdx.x;

    extern __shared__ float buffer[];

    //We reset the gradent and the error
    if(idx < n_points) { error[idx] = 0.0f; }

    if(idx < n_param) { grad[idx] = 0.0f; }

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
        //buffer[tdx] = 2 * error[idx] * d_X[idx * n_param + k];

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
__global__ void updateParameters(
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
    float learning_rate, float desired_tol, float mse
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

    //The main loop of the algorithm
    do{
        gradientDescent <<<numBlocks, numThreads, shared_mem>>>(
            X->data_d, y->data_d,
            parameters->data_d, gradient->data_d, error->data_d,
            n_points, n_param
        );

        updateParameters <<<1, numThreadsParameters>>>(
            parameters->data_d, gradient->data_d, 
            learning_rate, n_points, n_param
        );

    }while(
        (++iter < n_iter) &&
        (checkError(iter, desired_tol, &mse, &learning_rate, error))    
    );

    cudaDeviceSynchronize();

    //We bring back only the parameters
    copyMemory(parameters, DEVICE_TO_HOST);
}


//Every 10 iterations we check if the tolerance is met
__host__ bool checkError(int iter, float desired_tol, float* mse, float* learning_rate, tensor* error){
    
    if(iter % 10 == 0){

        //We use an auxiliary variable to compare with the value of the previous iteration so that
        //we can catch when the gradent "bounces" back
        float mse_aux = calculateNorm(error);

        if (
            (*mse < mse_aux) &&
            (iter != 10) //For not reducing the learning rate at the beggining
        ){
            *learning_rate = 0.1 * *learning_rate;

            if(*learning_rate <= 1e-10) {return false;}

            std::cout << "Se ha cambiado la tasa de aprendizaje en la " << iter << " iteracion. Alpha = "<< *learning_rate << std::endl;
        }

        *mse = mse_aux;

        if(*mse <= desired_tol){
            std::cout << "Se ha alcanzado la tolerancia esperada a las " << iter << " iteraciones\n" << std::endl;
            return false;
        }

        return true;

    }else{
        return true;
    }
}


//A first approach to calculate the euclidean norm of a vector in CPU
__host__ float calculateNorm(tensor* vector){
    float norm = 0.0f;

    cudaDeviceSynchronize();
    copyMemory(vector, DEVICE_TO_HOST);

    for(int i = 0; i < vector->rows; ++i){
        norm += vector->data_h[i] * vector->data_h[i];
    }

    return norm;
}