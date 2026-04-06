#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "lr/amdLinearRegression.h"

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
}