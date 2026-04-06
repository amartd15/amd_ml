#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "lr/amdLinearRegression.h"

namespace py = pybind11;

py::array_t<float> py_linear_regression(
    py::array_t<float> X,
    py::array_t<float> y,
    int n_iter,
    float tolerance,
    float seed,
    float learning_rate
){
    auto X_buf = X.request();
    auto y_buf = y.request();

    int n_points     = X_buf.shape[0];
    int n_parameters = X_buf.shape[1];

    amd_linear_regression model = linear_regression(
        (float*)X_buf.ptr, (float*)y_buf.ptr,
        n_points, n_parameters, n_iter,
        tolerance, seed, learning_rate,
        NO_BIAS
    );

    // Copiamos los parámetros a un array de Numpy para devolverlos
    py::array_t<float> result(model.parameters->rows);
    auto res_buf = result.request();
    memcpy(res_buf.ptr, model.parameters->data_h,
           model.parameters->rows * sizeof(float));

    cleanContext(model);
    return result;
}

PYBIND11_MODULE(amd_ml_py, m){
    m.doc() = "Linear Regression CUDA";

    m.def("linear_regression", &py_linear_regression,
        py::arg("X"),
        py::arg("y"),
        py::arg("n_iter")          = 1000,
        py::arg("tolerance")       = 0.0001f,
        py::arg("seed")            = 1.0f,
        py::arg("learning_rate")   = 0.1f
    );
}