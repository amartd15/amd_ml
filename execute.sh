#!/bin/bash

echo "-------> ELIMINANDO BUILD"
rm -rf build

echo "------> CREANDO BUILD"
cmake -B build -DCMAKE_BUILD_TYPE=Debug

echo "------> COMPILANDO"
cmake --build build

# Exportamos la ruta de las librerías .so para que el ejecutable las encuentre
export LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH

#echo "------> EJECUTANDO REGRESION LINEAL"
#cuda-gdb ./build/amd_ml_lr

#echo "------> EJECUTANDO REGRESION LINEAL SGD"
#./build/amd_ml_SGD