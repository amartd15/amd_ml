#include "amdMemoryManagement.h"


//Creates an empty tensor with allocated Pinned and Device memory
__host__ tensor* createTensor(int rows, int cols){
    tensor* created_tensor = (tensor*)malloc(sizeof(tensor));

    created_tensor->columns = cols;
    created_tensor->rows = rows;

    size_t size = cols * rows * sizeof(float);

    created_tensor->data_h = allocatePinnedMemory(size);
    created_tensor->data_d = allocateDeviceMemory(size);

    return created_tensor;
};


//Creates a tensor with the given data both in device and host memory
__host__ tensor* createTensor(float* data, int rows, int cols){
    tensor* created_tensor = (tensor*)malloc(sizeof(tensor));

    created_tensor->columns = cols;
    created_tensor->rows = rows;

    size_t size = cols * rows * sizeof(float);

    created_tensor->data_h = data;
    created_tensor->data_d = allocateDeviceMemory(size);

    copyMemory(created_tensor, HOST_TO_DEVICE);

    return created_tensor;
};


//Creates a tensor with the given seed both in device and host memory
__host__ tensor* createTensor(float seed, int rows, int cols){
    tensor* created_tensor = (tensor*)malloc(sizeof(tensor));

    created_tensor->columns = cols;
    created_tensor->rows = rows;

    size_t size = cols * rows * sizeof(float);

    created_tensor->data_h = allocatePinnedMemory(size);
    created_tensor->data_d = allocateDeviceMemory(size);

    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            created_tensor->data_h[j + i*cols] = seed;
        }
    }

    copyMemory(created_tensor, HOST_TO_DEVICE);

    return created_tensor;
};


__host__ BTCH_tensor* BTCHcreateTensor_y(float* data, int n_points, int BTCH){
    BTCH_tensor* created_tensor = (BTCH_tensor*)malloc(sizeof(BTCH_tensor));

    size_t size = BTCH * sizeof(float);

    //For creating pinned memory
    CUDA_CHECK(
        cudaHostRegister((void*) data, n_points * sizeof(float), cudaHostRegisterDefault),
        "Creating pinned memory"
    );

    created_tensor->data_h = data;

    //For both device buffers
    for(int i = 0; i < 2; ++i){
        CUDA_CHECK(
            cudaMalloc((void**) &created_tensor->data_d[i], size), 
            "Creating pinned memory for tensor points"
        );
    };

    return created_tensor; 
}


//Allocates memory in RAM
__host__ float* allocatePinnedMemory(size_t size){
    
    float* ptr;

    CUDA_CHECK(cudaMallocHost((void**)&ptr, size), "Allocating pinned memory");

    return ptr;
};


//Allocates mamory in VRAM (GPU)
__host__ float* allocateDeviceMemory(size_t size){
    
    float* ptr;

    CUDA_CHECK(cudaMalloc((void**)&ptr, size), "Allocating device memory");

    return ptr;
};


//Copy memory in any given direction between RAM and VRAM
__host__ void copyMemory(tensor* data, direction direction){

    size_t size = data->rows * data->columns * sizeof(float);

    if(direction == HOST_TO_DEVICE){
        CUDA_CHECK(cudaMemcpy(data->data_d, data->data_h, size, cudaMemcpyHostToDevice), "Copying memory from hot to device");

    }else{
        CUDA_CHECK(cudaMemcpy(data->data_h, data->data_d, size, cudaMemcpyDeviceToHost), "Copying memory form device to host");

    }
};


//Free memory from both RAM and VRAM
__host__ void freeTensor(tensor* data){

    CUDA_CHECK(cudaFree(data->data_d), "Free device memory from tensor");

    CUDA_CHECK(cudaFreeHost(data->data_h), "Free host memory from tensor");

    free(data);
};


//Free memory from both RAM and VRAM
__host__ void freeTensor(tensor* data, std::string msg){

    CUDA_CHECK(cudaFree(data->data_d), msg);

    CUDA_CHECK(cudaFreeHost(data->data_h), msg);

    free(data);
};


//Free memory only from device, and free the structure
__host__ void freeTensor(tensor* data, bias decision){

    CUDA_CHECK(cudaFree(data->data_d), "Free device memroy from tensor");

    if(decision == YES_BIAS){
        CUDA_CHECK(cudaFreeHost(data->data_h), "Free pinned memory from tensor");
    }else{
        free(data);
    }
};


//Free memory only from device, and free the structure
__host__ void freeTensor(tensor* data, bias decision, std::string msg){

    CUDA_CHECK(cudaFree(data->data_d), msg);

    if(decision == YES_BIAS){
        CUDA_CHECK(cudaFreeHost(data->data_h), msg);
    }else{
        free(data);
    }
}


//In case of the points matrix, we preparate the tensor wether we want bias or not
__host__ tensor* preparePointsTenstor(float* point_matrix, int n_parameters, int n_points, bias decision){

    //We prepare the tensor for the points matrix
    tensor* matrix_points;

    //We prepare the data, taking into acount if the user wants a bias
    if(decision == YES_BIAS){

        //Create an auxiliary array for the matrix with bias
        float* point_matrix_bias = allocatePinnedMemory((n_parameters+1) * n_points * sizeof(float));

        //We copy the array to a new one with a the first column of 1's (bias) 
        for(int i = 0; i < n_points; ++i){
            point_matrix_bias[i * (n_parameters + 1)] = 1.0f;

            for(int j = 0; j < n_parameters; ++j){
                point_matrix_bias[(j+1) + i*(n_parameters + 1)] = point_matrix[j + i*n_parameters];
            }
        }

        //We take into account we have one parameter more
        ++n_parameters;

        matrix_points = createTensor(point_matrix_bias, n_points, n_parameters);
    }else{
        matrix_points = createTensor(point_matrix, n_points, n_parameters);
    }

    return matrix_points;
}


//For the batch descent
__host__ BTCH_tensor* BTCHpreparePointsTenstor(float* point_matrix, int n_parameters, int n_points, bias decision, int BTCH){

    //We prepare the tensor for the points matrix
    BTCH_tensor* matrix_points = (BTCH_tensor*)malloc(sizeof(BTCH_tensor));

    //We prepare the data, taking into acount if the user wants a bias
    if(decision == YES_BIAS){

        //Create an auxiliary array for the matrix with bias
        float* point_matrix_bias = allocatePinnedMemory((n_parameters+1) * n_points * sizeof(float));

        //We copy the array to a new one with a the first column of 1's (bias) 
        for(int i = 0; i < n_points; ++i){
            point_matrix_bias[i * (n_parameters + 1)] = 1.0f;

            for(int j = 0; j < n_parameters; ++j){
                point_matrix_bias[(j+1) + i*(n_parameters + 1)] = point_matrix[j + i*n_parameters];
            }
        }

        //We take into account we have one parameter more
        ++n_parameters;

        //Host buffer
        matrix_points->data_h  = point_matrix_bias;

        //Both device buffer
        for(int i = 0; i < 2; ++i){
            CUDA_CHECK(
                cudaMalloc((void**) &matrix_points->data_d[i], BTCH * n_parameters * sizeof(float)), 
                "Creating pinned memory for tensor points"
            );
        };
        
    }else{

        //We make pinned memory from the data
        CUDA_CHECK(
            cudaHostRegister((void*) point_matrix, n_points * n_parameters * sizeof(float), cudaHostRegisterDefault),
            "Creating pinned memory"
        );

        matrix_points->data_h  = point_matrix;

        //Both device buffers
        for(int i = 0; i < 2; ++i){
            CUDA_CHECK(
                cudaMalloc((void**) &matrix_points->data_d[i], BTCH * n_parameters * sizeof(float)), 
                "Creating pinned memory for tensor points"
            );
        };
    }

    return matrix_points;
}


//We clean up all the variables created in the program
__host__ void cleanContext(amd_linear_regression context){
    freeTensor(context.error, "Free error tensor");
    freeTensor(context.parameters, "Free parameter tensor");
    freeTensor(context.gradient, "Free gradent tensor");
    freeTensor(context.result_matrix, context.hiperparameters->decision, "Free X tensor");
    freeTensor(context.point_matrix, context.hiperparameters->decision, "Free y tensor");

    free(context.hiperparameters);
}


//Good practice to clean up the device
__host__ void cleanUpDevice(){
    CUDA_CHECK(cudaDeviceReset(), "Resetting the device");
}
