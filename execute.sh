#!/bin/bash

echo "-------> ELIMINANDO BUILD"
rm -rf build

echo "------> CREANDO BUILD"
cmake -B build

echo "------> COMPILANDO"
cmake --build build

echo "------> EJECUTANDO REGRESION LINEAL"
./build/amd_ml_lr

echo "------> EJECUTANDO REGRESION LINEAL POR SGD"
./build/amd_ml_SGD

echo "------> EJECUTANDO SCRIPT PYTHON LR"
python3 Benchmarks/benchmark_lr.py