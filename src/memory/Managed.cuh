#pragma once
#include <cuda_runtime.h>
#include "utility/Error.cuh"

class Managed {
public:
    void* operator new(size_t len) {
        void* ptr;
        checkCudaErrors(cudaMallocManaged(&ptr, len));
        cudaDeviceSynchronize();
        return ptr;
    }

    void operator delete(void* ptr) {
        cudaDeviceSynchronize();
        checkCudaErrors(cudaFree(ptr));
    }
};