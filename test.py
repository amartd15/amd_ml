import sys
sys.path.append("build")  # para que Python encuentre el .so

import amd_ml_py
import numpy as np

# Generamos datos de prueba — misma línea que tu main.cu
# y = 3*x + 2
n_puntos = 1000
X = np.zeros((n_puntos, 2), dtype=np.float32)
y = np.zeros(n_puntos, dtype=np.float32)

for i in range(n_puntos):
    X[i, 0] = 1.0
    X[i, 1] = i / n_puntos
    y[i] = 3.0 * X[i, 1] + 2.0 * X[i, 0]

params: np.ndarray = amd_ml_py.linear_regression(
    X, y,
    n_iter=1000,
    tolerance=0.01,
    learning_rate=0.1
)

print(f"Parámetros encontrados: {params.round(1)}")
print(f"Esperado: [2.0, 3.0]")