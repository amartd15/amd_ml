#include "SGD/kernelSGDLinearRegresion.h"
#include "amdMemoryManagement.h"

//Performs the calculation of the gradent, as well as clearing the gradient and error matrix
__global__ void StocasticGradientDescent(
    const float* d_X, const float* d_y, 
    float* param, float* grad,
    int n_points, int n_param, int iter, int n_iter
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

            grad[idx] = 2 * (h - d_y[row]) * d_X[row * n_param + idx];

            float alpha = 0.1 / (1 + 100 * row * iter / n_iter);
            param[idx] -= grad[idx] * alpha; 

            __syncthreads();

        }
    }

}


//This function encapsulates the process of launching the kernel of the linear regression.
//Only brings back to host memory the parameters matrix
__host__ void SGDlinearRregresionKernel(
    tensor* X, tensor* y, tensor* parameters, tensor* gradient, tensor* error,
    int n_param, int n_points, int n_iter, 
    float learning_rate, float desired_tol, float mse
){

    dim3 numThreads, numBlocks;

    //After some benchmarks the best result is given with 128 - 256 threads per block
    numThreads.x = 128;
    numThreads.y = 1;
    numThreads.z = 1;
    
    numBlocks.x = (int) (n_param + numThreads.x - 1) / numThreads.x;
    numBlocks.y = 1;
    numBlocks.z = 1;

    int iter = 0;

    //The main loop of the algorithm
    do{
            StocasticGradientDescent <<<numBlocks, numThreads>>>(
                X->data_d, 
                y->data_d,
                parameters->data_d, 
                gradient->data_d,
                n_points, n_param, iter, n_iter
            );

    }while(++iter < (int) n_iter);

    cudaDeviceSynchronize();

    //We bring back only the parameters
    copyMemory(parameters, DEVICE_TO_HOST);
}


//Every 10 iterations we check if the tolerance is met
__host__ bool checkError(int iter, float desired_tol, float* mse, float* learning_rate, tensor* error){
    
    if(iter % 10 == 0){

        //We use an auxiliary variable to compare with the value of the previous iteration so that
        //we can catch when the gradent "bounces" back
        *mse = calculateNorm(error);

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