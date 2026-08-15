#include "Btch/kernelBtchLinearRegresion.h"
#include "amdMemoryManagement.h"


#define BTCH 1000


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

    //We calculate the thread index
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tdx = threadIdx.x;

    //Define chared memory
    extern __shared__ float buffer[];

    //We create the function and calculate the error for each point
    if(idx < n_points){

        float h = 0.0f;

        #pragma unroll
        for(int k = 0; k < n_param; ++k){
            h += d_X[idx * n_param + k] * param[k];
        }

        error[idx] =  h - d_y[idx];

    };

    //We calculate the gradent for each parameter
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
}


//Updates de parameters in a different kernel
__global__ void lr_update_param(
    float* param, 
    float* grad, 

    float alpha, 

    int n_points, 
    int n_param
){

    //We calculate the thread index
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    //Update the parameters
    if(idx < n_param){
        param[idx] -= grad[idx] * alpha / n_points; 
    }
}


//-------------------------------------------------- LAUNCHING KERNEL -------------------------------------------//

//This function encapsulates the process of launching the kernel of the linear regression.
//Only brings back to host memory the parameters matrix, the rest is kept in device memory
__host__ amd_linear_regression BTCHlinearRregresionKernel(
    BTCH_tensor*        X, 
    BTCH_tensor*        y, 

    tensor*             parameters, 
    tensor*             gradient, 
    tensor*             error,
    
    lr_hiperparameters* hp //Short for hiperparameters
){

    int num_BTCH = (int) hp->n_points / BTCH;

    //---------------------Setting the kernel launching characteristics--------------------------//

    //We check wether there are more parameters or points in the dataset
    int length = (hp->n_param < hp->n_points) ? hp->n_points : hp->n_param;

    //We define the variables needed for launching the kernel
    dim3 numThreads, numBlocks, numBlocksParam;

    numThreads      = {64,                                                      1, 1}; //After some benchmarks the best result is given with 64 - 256 threads per block
    numBlocks       = {(int) (length + numThreads.x - 1) / numThreads.x,        1, 1};
    numBlocksParam  = {(int) (hp->n_param + numThreads.x - 1) / numThreads.x,   1, 1};

    //We define the shared memory we will be using
    int shared_mem  = numThreads.x * sizeof(float);

    //We create the control variables
    float  mse      = 0.0f; //To store each iteration the mse
    float* smse_aux = createSharedPointer(0.0f); //To calculate each iteration the mse

    float  mse_arr[num_BTCH];
    float  mse_arr_aux[num_BTCH]; 


    //---------------------Creating streams and events--------------------------//

    //We create the streams
    int n_streams = 3;
    cudaStream_t stream[n_streams];

    for(int i = 0; i < n_streams; ++i){
        CUDA_CHECK(cudaStreamCreate(&stream[i]), "Creating stream");
    }

    //We create an event
    cudaEvent_t computation_done;
    CUDA_CHECK(cudaEventCreate(&computation_done), "Creating event");


    //---------------------First iteration out of the loop--------------------------//

    //We copy the first time the data
    size_t offset = BTCH * sizeof(float);

    //Copying the batch in the points vector
    cudaMemcpyAsync(
        (void*) X->data_d[0], 
        (const void*) X->data_h, 

        //We pass BTCH points, but each one has its own set of coordinates
        offset * hp->n_param, 

        cudaMemcpyHostToDevice, 
        stream[0]
    );

    //Copying the batch in the y vector
    cudaMemcpyAsync(
        (void*) y->data_d[0], 
        (const void*) y->data_h, 

        //The first batch
        offset, 

        cudaMemcpyHostToDevice, 
        stream[0]
    );


    //---------------------Main loop--------------------------//

    //The main loop of the algorithm. Encharged of processing chinks of data
    for(int iter = 0; iter < hp->iter; ++iter){
        for(int chunk = 0; chunk < num_BTCH; ++chunk){

            //We will use one stream to upload memory and one for launching the kernel
            int stream_index_1 =  chunk      % n_streams;
            int stream_index_2 = (chunk + 1) % n_streams;

            //For switching between the two buffers
            int processed      = chunk       % 2;
            int transferred    = (chunk + 1) % 2;

            //Beware that it must be divisible or we are accessing ilegal memory adresses
            //We start with an offset to start in 1 offset (the other was done outside)
            int offset_index   = (chunk + num_BTCH + 1) % num_BTCH; 


            //---------------------Copying the data on one stream--------------------------//

            //For the last batch
            if(chunk == num_BTCH - 1){
                offset = hp->n_points - chunk*BTCH*sizeof(float);
            };

            //The memory transfers
            cudaMemcpyAsync(
                //We skip different number of batches depending on the iteration
                (void*) X->data_d[transferred],

                (void*) (X->data_h + 
                    offset_index * BTCH * hp->n_param), 

                //The fixed batch size
                offset * hp->n_param, 

                cudaMemcpyHostToDevice, 
                stream[stream_index_2]
            );

            cudaMemcpyAsync(
                (void*) y->data_d[transferred],

                (void*) (y->data_h + 
                    offset_index * BTCH), 

                //Fixed offset
                offset, 

                cudaMemcpyHostToDevice, 
                stream[stream_index_2]
            );


            //---------------------Copying the data on one stream--------------------------//

            //Reset the gradent
            cudaMemsetAsync(
                gradient->data_d, 
                0, 
                hp->n_param * sizeof(float), 
                stream[stream_index_1]
            );

            //Calculate the gradent
            lr_gradientDescent<<<numBlocks, numThreads, shared_mem, stream[stream_index_1]>>>(
                X->data_d[processed], 
                y->data_d[processed],
                parameters->data_d, 
                gradient->data_d, 
                error->data_d,
                BTCH, 
                hp->n_param
            );

            //Update the parameters
            lr_update_param<<<numBlocksParam, numThreads, 0, stream[stream_index_1]>>>(
                parameters->data_d, 
                gradient->data_d, 
                *hp->alpha, 
                BTCH, 
                hp->n_param
            );

            //We create an event
            cudaEventRecord(computation_done, stream[stream_index_1]);

            //We syncronize the event on the kernel stream
            cudaStreamWaitEvent(stream[stream_index_2], computation_done, cudaEventWaitDefault);

            //---------------------CAlculating error--------------------------//

            if(iter % hp->first_iteration == 0 && iter != hp->first_iteration){
                lr_norm(error, smse_aux);
                mse_arr[chunk] = *smse_aux;
            };
        }; //for(int chunk = 0; chunk < num_BTCH; ++chunk)

        if(iter % hp->first_iteration == 0){

            //We get a first value of mse_arr
            if(iter == hp->first_iteration){
                for(int i = 0; i < num_BTCH; ++i){
                    mse_arr_aux[i] = mse_arr[i];
                };   
            };

            if(!BTCH_compare_mse(&mse_arr[0], &mse_arr_aux[0], hp, num_BTCH)){ break; };
        };

    };//for(int iter = 0; iter < hp->iter; ++iter)
    

    //---------------------Cleanning the device, streams and events--------------------------//

    //We check for silent errors
    CUDA_CHECK(cudaGetLastError(), "Launching the kernels of the linear regression");

    //We syncronize de device
    CUDA_CHECK(cudaDeviceSynchronize(), "Device syncronization");  

    //Free shared pointers
    CUDA_CHECK(cudaFree(smse_aux), "Free smse_aux");

    //We get rid of the auxiliary buffer
    CUDA_CHECK(cudaFree(X->data_d[1]), "Freeing auxiliary buffer");
    CUDA_CHECK(cudaFree(y->data_d[1]), "Freeing auxiliary buffer");

    //Deleting the streams
    for(int i = 0; i < n_streams; ++i){
        CUDA_CHECK(cudaStreamDestroy(stream[i]), "Destroying stream");
    }

    //DEleting the event
    CUDA_CHECK(cudaEventDestroy(computation_done), "Destroying event");


    //---------------------Retrieveing the data--------------------------//

    //We bring back only the parameters
    copyMemory(parameters, DEVICE_TO_HOST);

    //We get the mse
    hp->mse                          = mse;

    //We create and fill the structure of the function
    amd_linear_regression context;

    context.point_matrix             = (tensor*) malloc(sizeof(tensor));
    context.result_matrix            = (tensor*) malloc(sizeof(tensor));

    //We "cast" the BTCH_tensor into tensors
    context.point_matrix->data_d     = X->data_d[0];
    context.point_matrix->data_h     = X->data_h;
    context.point_matrix->rows       = hp->n_points;
    context.point_matrix->columns    = hp->n_param;

    context.result_matrix->data_d    = y->data_d[0];
    context.result_matrix->data_h    = y->data_h;
    context.result_matrix->rows      = hp->n_points;
    context.result_matrix->columns   = 1;

    //We fill the context
    context.parameters               = parameters;
    context.error                    = error;
    context.gradient                 = gradient;
    
    context.hiperparameters          = hp;

    return context;
};


//-------------------------------------------------- CHECK TOLERANCES -------------------------------------------//

//Identifies if we had a bounce back
__host__ bool BTCH_compare_mse(
    float* mse_arr,
    float* mse_arr_aux,

    lr_hiperparameters* hp,
    int num_BTCH
){

    //We check the tolerance
    float mean_mse = 0;
    for(int i = 0; i < num_BTCH; ++i){
        mean_mse += mse_arr[i];
    };  

    mean_mse = mean_mse / num_BTCH;

    if(mean_mse < hp->tol){
        std::cout << "Tolerance met" << std::endl;
        return false;
    };    

    //We check how many times we had a bounce back
    int n_bounces = 0;

    for(int i = 0; i < num_BTCH; ++i){
        if(mse_arr_aux[i] - mse_arr[i] < 0){ 
            n_bounces++; 
            std::cout << "Bounce detected" << std::endl;
        };
    };  

    //If more than half of the batches had a bounce back, we reduce the learning rate
    if(n_bounces > num_BTCH / 2){
        *hp->alpha *= hp->alpha_reduction;
        std::cout << "REducing learning rate" << std::endl;

        if(*hp->alpha < hp->alpha_min){ return false; };
    };

    //Finally, we exchange values
    for(int i = 0; i < num_BTCH; ++i){
            mse_arr_aux[i] = mse_arr[i];
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

    //We set the memory to 0 blocking the GPU
    cudaMemsetAsync(mse_aux, 0, sizeof(float));
    
    //We launch the kernel
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
