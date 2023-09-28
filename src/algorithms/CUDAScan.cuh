#pragma once

#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "memory/UnifiedArray.cuh"
#include "utility/KernelLaunchParams.h"

#define THREAD_ID threadIdx.x + blockIdx.x * blockDim.x

#define CUDA_SCAN_THREADS 512

__host__ __device__ uint32_t log2i(uint32_t i);

__host__ __device__ uint32_t exp2i(uint32_t i);

__global__ void hills_steele_step(UnifiedArray<uint32_t>* p_in, UnifiedArray<uint32_t>* p_out, uint32_t i);

__global__ void copy(UnifiedArray<uint32_t>* p_in, UnifiedArray<uint32_t>* p_out);

void swapPointers(UnifiedArray<uint32_t>*& p_front, UnifiedArray<uint32_t>*& p_back);

void cudaScan(UnifiedArray<uint32_t>* p_in, UnifiedArray<uint32_t>* p_out);

