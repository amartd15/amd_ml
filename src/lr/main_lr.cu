#include "lr/amdLinearRegression.h"

__host__ void title(std::string mes){
    std::cout << "\n-----------------" << mes << "-----------------\n" << std::endl;
}

__host__ void imprimir(tensor* inp, std::string name){
    std::cout << "\nPrinting amd_lr_tensor " << name << std::endl;

    for (int i=0; i < inp->rows; ++i){
        for(int j=0; j < inp->columns; ++j){
            printf("%.1f ", inp->data_h[j + i*inp->columns]);
        }
        printf("\n");
    }

    printf("\n");
}

__host__ int main(){
    //Create the matrix of coeficients and the matrix of points

    title("Creating parameters");

    int n_parametros = 2;
    int n_puntos = 1000;

    float m = 3.0f;
    float n = 2.0f;

    float seed = 1;
    float learning_rate = 0.1;
    float iterations = 1000;
    float tol_required = 0.003;

    float gaussian_blurr = 0.1f;

    std::default_random_engine generator;
    std::normal_distribution<float> custom_gaussian_random(0, 1);

    float* X = (float*)malloc(n_parametros * n_puntos * sizeof(float));
    float* y = (float*)malloc(n_puntos * sizeof(float));

    title("Filling up amd_lr_tensors");

    for(int i=0; i < n_puntos; ++i){
        X[i * n_parametros] = 1.0f;
        X[i*n_parametros + 1] = (float)i / n_puntos;
        y[i] = m * X[i * n_parametros + 1]
                     + n * X[i * n_parametros] 
                     + custom_gaussian_random(generator) * gaussian_blurr;
    }

    title("running kernel");

    cudaEvent_t a, b;
    float ms = 0;

    cudaEventCreate(&a);
    cudaEventCreate(&b);

    cudaEventRecord(a);

    amd_linear_regression model = linear_regression(
        X, y, 
        n_puntos, n_parametros, iterations, 
        tol_required, seed, learning_rate, 
        NO_BIAS
    );

    cudaEventRecord(b);
    cudaEventSynchronize(b);

    cudaEventElapsedTime(&ms, a, b);

    cudaEventDestroy(a);
    cudaEventDestroy(b);

    imprimir(model.parameters, "Parameters");

    //std::cout << "-->MSE (mean squared error): " << model.mse / n_puntos << std::endl;
    std::cout << "-->El proceso tardo " << ms<< " milisegundos\n" << std::endl;

    cleanContext(model);
    cleanUpDevice();

    return 0;
};

