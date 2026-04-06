HEADERS = -I"include"
COMPILER = nvcc
FLAGS = -std=c++17 #-g -G -O0
SOURCE = $(shell find src -name '*.cu')

all: clean build run

build:
	$(COMPILER) $(SOURCE) $(FLAGS) $(HEADERS) -o bin/amd_ml.exe -lcuda

clean:
	rm -r bin/*.exe

run:
	./bin/amd_ml.exe

.PHONY: all clean build run