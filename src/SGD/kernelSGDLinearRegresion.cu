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
    int n_points, int n_param, float learning_rate
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
__host__ void SGDlinearRregresionKernel(
    tensor* X, tensor* y, tensor* parameters, tensor* gradient, tensor* error,
    unsigned int n_param, unsigned int n_points, unsigned int n_iter, 
    float learning_rate, float desired_tol
){

    //We check wether there are more parameters or points in the dataset
    int length = (n_param < n_points) ? n_points : n_param;

    //We define the variables needed for launching the kernel
    dim3 numThreads, numBlocks, numBlocksError;

    numThreads = {128, 1, 1}; //After some benchmarks the best result is given with 128 - 256 threads per block
    numBlocks = {(int) (length + numThreads.x - 1) / numThreads.x, 1, 1};

    numBlocksError = {(int) (n_param + numThreads.x - 1) / numThreads.x, 1, 1};

    int iter = 0;
    tensor* mse = createTensor(0.0f, 1, 1); //For tracking if the model bounces back


    //The main loop of the algorithm
    do{
            StocasticGradientDescent <<<numBlocks, numThreads>>>(
                X->data_d, 
                y->data_d,
                parameters->data_d,
                n_points, n_param, learning_rate
            );

            SGD_calculateError<<<numBlocksError, numThreads>>>(
                X->data_d,
                y->data_d,
                parameters->data_d,
                error->data_d,
                n_points,
                n_param
            );

    }while(
        (++iter < n_iter) &&
        (SGD_checkError(iter, desired_tol, mse, &learning_rate, error))    
    );

    cudaDeviceSynchronize();

    //We check for silent errors
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cout << "Error during the kernel launch. Error message -> " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    //We bring back only the parameters
    copyMemory(parameters, DEVICE_TO_HOST);

    //We free the pointer used for tracking the mse
    freeTensor(mse);
}


//-------------------------------------------------- CHECK TOLERANCES -------------------------------------------//


//We calculate the error with our current model with the dataset
__global__ void SGD_calculateError(
    const float* d_X,
    const float* d_y,
    const float* param,
    float* error,
    int n_points, int n_param
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
__host__ bool SGD_checkError(int iter, float desired_tol, tensor* mse, float* alpha, tensor* error){
    
    if(iter % ITERATION_CHECK_N == 0){

        //We use an auxiliary variable to compare with the value of the previous iteration so that
        //we can catch when the gradent "bounces" back

        float mse_aux = SGD_calculateNorm(error);

        //If we detect a bounce back, we modify the learning rate
        if (
            (*mse->data_h < mse_aux) &&
            (iter != ITERATION_CHECK_N) //For not reducing the learning rate at the beggining
        ){
            *alpha *= LEARNING_RATE_REDUCTION; //Reduce the learning rate

            if(*alpha <= MINIMUM_LEARNING_RATE) {return false;}

            std::cout << "Se ha cambiado la tasa de aprendizaje en la " << iter << " iteracion. Alpha = "<< *alpha << std::endl;
        }

        //We swap the values
        *mse->data_h = mse_aux;

        //We check if the tolerance is met
        if(*mse->data_h <= desired_tol){
            std::cout << "Se ha alcanzado la tolerancia esperada a los " << iter << " barridos\n" << std::endl;
            return false;
        }

        return true;

    }else{
        return true;
    }
}


//Encapsulates the launch of a kernel that calculates the euclidean norm of an horizontal or vertical vector
__host__ float SGD_calculateNorm(tensor* vector){

    //An auxiliary variable initalized with 0 values
    tensor* mse_squared = createTensor(0.0f, 1, 1);

    int size;

    //We calculate wether the vector is a row or a column
    if(min(vector->rows, vector->columns) == 1){
        size = max(vector->rows, vector->columns);
    }else{
        std::cout << "Error, trying to calculate the norm of a matrix" << std::endl;
    }

    //Some parameters to launch the kernel
    dim3 numThreads = {128, 1, 1};
    dim3 numBlocks  = {(int) (size + numThreads.x -1) / numThreads.x, 1, 1};

    //Shared memory
    int sharedMem = numThreads.x * sizeof(float);

    //We launch the kernel
    SGD_norm<<<numBlocks, numThreads, sharedMem>>>(vector->data_d, mse_squared->data_d, size);

    //We check for silent errors during the kernel launch
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cout << "Error launching the error kernel" << std::endl;
        exit(EXIT_FAILURE);
    }

    //Bring back the mse squared to host memory and storeing it on a variable
    copyMemory(mse_squared, DEVICE_TO_HOST);
    float value = *mse_squared->data_h;

    //To avoid memory leaks (on each iteration allocate memory and not cleaning it)
    freeTensor(mse_squared);

    return sqrt(value);
}


//Performs the euclidean norm of a vactor in GPU
__global__ void SGD_norm(float* data, float* value, int size){
    
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
    if(tdx == 0) {atomicAdd(value, buffer[0]);}

}