CC=g++
NVCC=nvcc
CXXFLAGS=-std=c++11 -O3

all: bench

bench: bench.cu  xxhash.hcu utils.hcu cuckoo_serial.hcu cuckoo_cuda_native.hcu
	${NVCC} $< -o $@ ${CXXFLAGS}

# demo: demo.cu cuckoo-serial.hpp cuckoo-cuda.cuh
# 	${NVCC} $< -o $@ ${CXXFLAGS}

.PHONY: clean
clean:
	rm -f bench 
