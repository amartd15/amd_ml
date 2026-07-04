#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "lr/amdLinearRegression.h"
#include "SGD/amdSGDLinearRegression.h"

namespace py = pybind11;

//WE create a function wrappet
py::array_t<float> py_linear_regression(
    py::array_t<float> X,
    py::array_t<float> y,
    int n_iter,
    float tolerance,
    float seed,
    float learning_rate
){
    //Request getting the numpy arrays as data we can access
    auto X_buf = X.request();
    auto y_buf = y.request();

    // DEBUG
    float* X_ptr = (float*)X_buf.ptr;
    float* y_ptr = (float*)y_buf.ptr;
    std::cout << "X[0,0]=" << X_ptr[0] << " X[0,1]=" << X_ptr[1] << std::endl;
    std::cout << "X[1,0]=" << X_ptr[X_buf.shape[1]] << " X[1,1]=" << X_ptr[X_buf.shape[1]+1] << std::endl;
    std::cout << "y[0]=" << y_ptr[0] << " y[1]=" << y_ptr[1] << std::endl;
    std::cout << "n_points=" << X_buf.shape[0] << " n_param=" << X_buf.shape[1] << std::endl;

    int n_points     = X_buf.shape[0];
    int n_parameters = X_buf.shape[1];

    //The X_buf.ptr return *void, we have to cast it
    amd_linear_regression model = linear_regression(
        (float*)X_buf.ptr, (float*)y_buf.ptr,
        n_points, n_parameters, n_iter,
        tolerance, seed, learning_rate,
        NO_BIAS
    );

    //We create the array to return, and request to use it
    py::array_t<float> result(model.parameters->rows);
    auto res_buf = result.request();

    //We pass the data to the ndarray we requested to use
    memcpy(res_buf.ptr, model.parameters->data_h,
           model.parameters->rows * sizeof(float));

    //Errase all data except the pointers we receive because they're not "ours"
    cleanContext(model);
    return result;
}


py::array_t<float> py_SGD_linear_regression(
    py::array_t<float, py::array::c_style | py::array::forcecast> X,
    py::array_t<float, py::array::c_style | py::array::forcecast> y,
    int n_iter,
    float tolerance,
    float seed,
    float learning_rate
){

    //Request getting the numpy arrays as data we can access
    auto X_buf = X.request();
    auto y_buf = y.request();

    int n_points     = X_buf.shape[0];
    int n_parameters = X_buf.shape[1];

    // y puede venir como (n,1) desde numpy — nos aseguramos de que tenga n elementos
    if(y_buf.ndim == 2 && y_buf.shape[1] == 1){
        // ya está C-contiguous, el ptr apunta a los datos correctos
        // pero comprobamos que n_points coincide
        if(y_buf.shape[0] != n_points){
            throw std::runtime_error("y debe tener el mismo número de filas que X");
        }
    } else if(y_buf.ndim != 1 || y_buf.shape[0] != n_points){
        throw std::runtime_error("y debe ser un vector de shape (n_points,) o (n_points, 1)");
    }

    //The X_buf.ptr return *void, we have to cast it
    amd_linear_regression model = SGD_linear_regression(
        (float*)X_buf.ptr, (float*)y_buf.ptr,
        n_points, n_parameters, n_iter,
        tolerance, seed, learning_rate,
        NO_BIAS
    );

    //We create the array to return, and request to use it
    py::array_t<float> result(model.parameters->rows);
    auto res_buf = result.request();

    //We pass the data to the ndarray we requested to use
    memcpy(res_buf.ptr, model.parameters->data_h,
           model.parameters->rows * sizeof(float));

    //Errase all data except the pointers we receive because they're not "ours"
    cleanContext(model);
    return result;
}

//Create the library
PYBIND11_MODULE(amd_ml_py, m){
    m.doc() = "Linear Regression CUDA";

    //One of the functions of the library, with its default arguments
    m.def("linear_regression", &py_linear_regression,
        py::arg("X"),
        py::arg("y"),
        py::arg("n_iter")          = 1000,
        py::arg("tolerance")       = 0.0001f,
        py::arg("seed")            = 1.0f,
        py::arg("learning_rate")   = 0.1f
    );

    m.def("SGD_linear_regression", &py_SGD_linear_regression,
        py::arg("X"),
        py::arg("y"),
        py::arg("n_iter")          = 1000,
        py::arg("tolerance")       = 0.0001f,
        py::arg("seed")            = 2.0f,
        py::arg("learning_rate")   = 0.1f
    );
}