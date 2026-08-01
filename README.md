# MACHINE LEARNING PROJECT - amd_ml

GPU-accelerated Linear Regression from scratch in CUDA — a from-first-principles implementation of batch gradient descent and stochastic gradient descent, with a Python binding so it can be dropped into a normal NumPy / scikit-learn workflow.

This started as a project to actually understand the principles of the linear regression models as well as testing adquired knowledge about CUDA. This project intended to serve as a programming trainning exercise, manually managing memory on the GPU, writing the reduction kernels by hand, and seeing how the result stacks up against scikit-learn on real hardware.

## What it does

- Implements **linear regression via gradient descent** and **SGD**, both fully in custom CUDA kernels (no cuBLAS).
- Handles all host ↔ device memory management manually (pinned host memory, device allocation, transfers, cleanup).
- Optional bias (intercept) term, added automatically to the design matrix if requested.
- Adaptive learning rate: detects when the loss "bounces back" between iterations and shrinks the learning rate automatically.
- Exposes the C++/CUDA core to Python through **pybind11**, so it can be called directly on NumPy arrays.
- Includes benchmark scripts that compare it against `sklearn.linear_model.LinearRegression` and `SGDRegressor` across dataset sizes and dimensionality.

## Requirements

- CUDA Toolkit 12.5 (or adjust `CMAKE_CUDA_COMPILER` in `CMakeLists.txt` to match your install)
- CMake ≥ 3.18
- A C++17 compiler
- Python3 with `pybind11` installed (for the Python bindings)
- For benchmarking: `numpy`, `scikit-learn`, `matplotlib`

## Building

The quickest way to build and run the plain gradient-descent executable:

```bash
./execute.sh
```

This wipes any previous `build/` directory, configures the project with CMake, compiles everything, and runs the linear regression demo (`build/amd_ml_lr`).

To build manually, including the Python module:

```bash
cmake -B build
cmake --build build

export LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH
./build/amd_ml_lr
./build/amd_ml_SGD
```

Both the `LR` and `SGD` targets can be toggled independently via the `LR` / `SGD` options in `CMakeLists.txt`.

## Using it from Python

Once built, `amd_ml_py` is importable from the `build/` directory:

```python
import sys
sys.path.append("build")

from amd_ml_py import linear_regression, SGD_linear_regression
import numpy as np

X = np.random.rand(1000, 2).astype(np.float32)
y = (3 * X[:, 1] + 2).astype(np.float32)

params = linear_regression(X, y, n_iter=1000, tolerance=1e-4, learning_rate=0.1)
print(params)
```

## Benchmarks

`Benchmarks/benchmark_lr.py` and `Benchmarks/benchmark_SGD.py` sweep over dataset size (log scale) and number of dimensions, timing `amd_ml` against the equivalent scikit-learn estimator, and plot the results in 3D (`lr_comparison_log.png`, `SGD_comparison_log.png`). Run them from the project root so they can find the compiled module:

```bash
python Benchmarks/benchmark_lr.py
python Benchmarks/benchmark_SGD.py
```

## How the kernels work (short version)

- Each thread computes the prediction error for one data point.
- A shared-memory tree reduction sums up the per-parameter gradient within each block, and `atomicAdd` combines the per-block partial sums into the global gradient.
- A second, small kernel applies the parameter update.
- Every `ITERATION_CHECK_N` iterations, the current error norm is compared against the previous one; if the loss increased ("bounced back"), the learning rate is reduced automatically until it either recovers or hits a minimum floor.

## Known limitations

This is a learning project, not a production ML library — a few things to be aware of:

- Only single precision (`float`) is supported.
- The Python bindings currently expose `linear_regression` and `SGD_linear_regression` without the bias/intercept option (`NO_BIAS` is hardcoded).
- No automated test suite yet; correctness has mainly been checked against scikit-learn output in the benchmark scripts.

## Debugging

If you want to step through the CUDA kernels or the `.so` bindings in VS Code, `DEBUGGING_GUIDE.md` walks through setting up `launch.json`/`settings.json` for CMake + CUDA + shared-library projects.

## License

No license has been set yet — treat this as source-available for now; reach out if you'd like to use it under specific terms.