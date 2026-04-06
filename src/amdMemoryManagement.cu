#include "amdMemoryManagement.h"


__host__ tensor* createTensor(int rows, int cols){
    tensor* created_tensor = (tensor*)malloc(sizeof(tensor));

    created_tensor->columns = cols;
    created_tensor->rows = rows;

    size_t size = cols * rows * sizeof(float);

    created_tensor->data_h = allocatePinnedMemory(size);
    created_tensor->data_d = allocateDeviceMemory(size);

    return created_tensor;
};


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


__host__ float* allocatePinnedMemory(size_t size){
    cudaError_t err;
    float* ptr;

    err = cudaMallocHost((void**)&ptr, size);
    if(err != cudaSuccess){
        std::cout << "An error ocurred allocating pinned memory. " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
        
    }

    return ptr;
};


__host__ float* allocateDeviceMemory(size_t size){
    cudaError_t err;
    float* ptr;

    err = cudaMalloc((void**)&ptr, size);
    if(err != cudaSuccess){
        std::cout << "Error allocating device memory. " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);

    }

    return ptr;
};


__host__ void copyMemory(tensor* data, direction direction){
    cudaError_t err;
    size_t size = data->rows * data->columns * sizeof(float);

    if(direction == HOST_TO_DEVICE){
        err = cudaMemcpy(data->data_d, data->data_h, size, cudaMemcpyHostToDevice);
        if(err != cudaSuccess){
            std::cout << "Error copying memory from host to device. " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);

        }

    }else{
        err = cudaMemcpy(data->data_h, data->data_d, size, cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            std::cout << "Error copying memory from device to host. " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);

        }

    }
};


__host__ void freeTensor(tensor* data){
    cudaError_t err;

    err = cudaFree(data->data_d);
    if(err != cudaSuccess){
        std::cout << "Error freeing device memory from the tensor. " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE); 

    }

    err = cudaFreeHost(data->data_h);
    if(err != cudaSuccess){
        std::cout << "Error freeing host memory from the tensor. " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE); 

    }

    free(data);
};


__host__ void freeTensor(tensor* data, bias decision){
    cudaError_t err;

    err = cudaFree(data->data_d);
    if(err != cudaSuccess){
        std::cout << "Error freeing memory from the tensor. " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE); 

    }

    //If we are taking pointers of python, we dont have to get rid of the host memory

    // if(decision == NO_BIAS){
    //     free(data->data_h);

    // }else{
    //     err = cudaFreeHost(data->data_h);
    //     if(err != cudaSuccess){
    //         std::cout << "Error freeing pinned memory from the tensor. " << cudaGetErrorString(err) << std::endl;
    //         exit(EXIT_FAILURE); 

    //     }
    // }

    free(data);
};


__host__ tensor* preparePointsTenstor(float* point_matrix, int n_parameters, int n_points, bias decision){

    //We prepare the tensor for the points matrix
    tensor* matrix_points;

    //We prepare the data, taking into acount if the user wants a bias
    if(decision == YES_BIAS){
        //Create an auxiliary array for the matrix with bias
        //float* point_matrix_bias = (float*)malloc((n_parameters+1) * n_points * sizeof(float));
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


__host__ void cleanContext(amd_linear_regression context){
    freeTensor(context.error);
    freeTensor(context.parameters);
    freeTensor(context.gradient);
    freeTensor(context.result_matrix, context.decision);
    freeTensor(context.point_matrix, context.decision);
}


__host__ void cleanUpDevice(){
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess){
        printf("--------------------Failed to deinitialize the device. Error: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}
