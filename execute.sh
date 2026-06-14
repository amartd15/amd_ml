#!/bin/bash
echo "Borrando la carpeta build"

rm -rf build

echo "Compilando" 

cmake -B build
cmake --build build