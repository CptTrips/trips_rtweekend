#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "visibles/CUDAVisible.cuh"
#include "visibles/CUDASphere.cuh"
#include <curand_kernel.h>
#include "materials/metal.cuh"
#include "materials/diffuse.cuh"
#include "materials/dielectric.cuh"
#include "rand.h"

#define seed 1234

CUDAVisible** single_ball();

__global__ void gen_single_ball(CUDAVisible** const scenery);

CUDAVisible** random_balls(const int ball_count);

__global__ void gen_random_balls(CUDAVisible** const scenery, const int ball_count);
