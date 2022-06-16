#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "visibles/CUDAVisible.cuh"
#include "visibles/CUDASphere.cuh"
#include "visibles/Triangle.cuh"
#include "visibles/Mesh.cuh"
#include <curand_kernel.h>
#include "materials/metal.cuh"
#include "materials/diffuse.cuh"
#include "materials/dielectric.cuh"
#include "rand.h"
#include "Error.cuh"
#include "CUDAScene.cuh"

#define my_cuda_seed 1234


__global__ void create_scene_buffers(CUDAScene* scenery, const int visible_count, const int material_count);


CUDAScene* single_ball();

__global__ void gen_single_ball(Array<CUDAVisible*>* visibles, Array<Material<CUDA_RNG>*>* materials);


CUDAScene* random_balls(const int ball_count);

__global__ void gen_random_balls(CUDAScene* const scene, const int ball_count);


CUDAScene* single_triangle();

__global__ void gen_single_triangle(CUDAScene* const scenery);


CUDAScene* single_cube();

__global__ void gen_single_cube(CUDAScene* const scenery);


void teardown_scene(CUDAScene* scene);

__global__ void cuda_teardown_scene(CUDAScene* scene);