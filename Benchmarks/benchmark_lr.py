import sys

from pathlib import Path
from timeit import time

#Tratamos de importar la librería con la que trabajamos
try:
    build = Path(__file__).resolve().parent.parent / "build" #Teniendo en cuenta que se ejecuta desde la carpeta base del proyecto
    print("\nBuscando librerias en -> ", build)

    sys.path.append(str(build))  # para que Python encuentre el .so

    from amd_ml_py import linear_regression
except Exception as e:
    print("No se pudo enlazar con la libreria externa-> ", e)
    exit()

#Importamos los otros paquetes, asegurándonos que el entorno virtual está activado
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
except ModuleNotFoundError:
    print("No se activo debidamente el entorno virtual o hay un problema con las librerias externas")
    exit()



def create_matrix(n_points: int, n_dimentions: int) -> list[np.ndarray]:
    #We generate a linear regression with coefficients 1, 2, 3, ...
    #We count with the bias term

    terms = np.zeros(n_dimentions + 1, dtype=np.float32)
    terms[0] = 1

    for i in range(n_dimentions):
        terms[i+1] = i+2

    X = np.ones((n_points, 1), dtype = np.float32)

    for i in range(n_dimentions):
        X = np.hstack((
            X, 
            np.linspace(0, 1, n_points).reshape(-1, 1) + np.random.randn(n_points).reshape(-1, 1)
            ))
    
    y = np.matmul(X, terms.reshape(-1, 1))

    return [X, y, terms]

def comparation(X: np.ndarray, y: np.ndarray, terms: np.ndarray, print_data = True) -> list[float]:
    a = time.time()
    parameters_sk = LinearRegression(fit_intercept=False).fit(X, y)
    b = time.time()

    c = time.time()
    params: np.ndarray = linear_regression(
        X, y,
        n_iter=1000,
        tolerance=0.01 * len(X),
        learning_rate=0.1
    )
    d = time.time()

    time_amd = d-c
    time_sk = b-a

    if print_data:
        print(f"Parámetros encontrados: {params.round(1)}, tiempo usado: {time_amd} milisegundos")
        print(f"Parámetros encontrados sklearn: {parameters_sk.coef_.round(1)}, tiempo usado: {time_sk} milisegundos")
        print(f"Esperado: {terms}")

    return [time_amd, time_sk]    
    
def comparation_amd_sk(n_points: int, n_dim: int) -> None:
    #Because of the compiler JIT, we initialice the memory
    comparation(*create_matrix(1000, 2), print_data=False)

    #We iterate through out powers of 10 in terms of points and through dimentions
    #points     = np.array([1000 * i for i in range(1, 1+n_points)], dtype=int)
    points     = np.array([10**i for i in range(1, 1+n_points)], dtype=int)
    dimentions = np.array([i+1 for i in range(n_dim)], dtype=int) 

    points_amd   = np.zeros((3, n_points * n_dim))
    points_sk    = np.zeros_like(points_amd)

    idx = 0

    for i in range(n_points):
        for j in range(n_dim):

            points_amd[0, idx] = points[i]
            points_amd[1, idx] = dimentions[j]

            points_sk[0, idx] = points[i]
            points_sk[1, idx] = dimentions[j]

            points_amd[2, idx], points_sk[2, idx] = comparation(*create_matrix(points[i], dimentions[j]), print_data=False)

            idx += 1

    #fig = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')

    ax.scatter(np.log10(points_amd[0, :]), points_amd[1, :], points_amd[2, :], label = "amd time")
    ax.scatter(np.log10(points_sk[0, :]) , points_sk[1, :] , points_sk[2, :] , label = "sk time")

    ax.set_xlabel("Number of points (logarithmic)")
    ax.set_ylabel("Number of dimentions")
    ax.set_zlabel("Time (miliseconds)")

    ax.legend()
    ax.set_title("Comparison between hardcoded and library")

    plt.show()

def main() -> None:
    comparation(*create_matrix(1000, 2))
    #comparation_amd_sk(5, 2)

if __name__ == "__main__":
    main()