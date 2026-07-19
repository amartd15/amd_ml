#!/bin/bash

echo "-------> ELIMINANDO BUILD"
rm -rf build

echo "------> CREANDO BUILD"
cmake -B build

echo "------> COMPILANDO"
cmake --build build

echo "------> EJECUTANDO REGRESION LINEAL"
./build/amd_ml_lr

echo "------> EJECUTANDO REGRESION LINEAL SGD"
./build/amd_ml_SGD