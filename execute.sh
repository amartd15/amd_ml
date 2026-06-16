#!/bin/bash

rm -rf build
cmake -B build
cmake --build build
python3 Benchmarks/benchmark_SGD.py