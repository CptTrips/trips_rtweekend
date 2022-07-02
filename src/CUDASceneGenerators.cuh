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
#include "UnifiedArray.cuh"

#define my_cuda_seed 1234



CUDAScene* scene_factory(const int visible_count, const int material_count);

CUDAScene* single_ball();

__global__ void gen_single_ball(CUDAScene* const scene);


CUDAScene* random_balls(const int ball_count);

__global__ void gen_random_balls(CUDAScene* const scene, const int ball_count);


CUDAScene* single_triangle();

__global__ void gen_single_triangle(CUDAScene* const scene);

Array<vec3>* cube_vertices(const vec3& translation);

Array<uint32_t>* cube_indices();


CUDAScene* single_cube();

__global__ void gen_single_cube(CUDAScene* const scene, const Array<vec3>* const vertex_array, const Array<uint32_t>* const index_array, Material<CUDA_RNG>* const mat);

CUDAScene* n_cubes(const int& n);

__global__ void gen_n_cubes(CUDAScene* const scene);

CUDAScene* triangle_carpet(const unsigned int& n);

__global__ void gen_carpet(CUDAScene* const scene);

void teardown_scene(CUDAScene* scene);

