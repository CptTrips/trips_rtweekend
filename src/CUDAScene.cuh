#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "materials\material.h"
#include "visibles\CUDAVisible.cuh"
#include "rand.h"
#include "Array.cuh"
#include "Managed.cuh"
#include "Error.cuh"


class CUDAScene : public Managed
{
public:
	__host__ CUDAScene();
	__host__ CUDAScene(Array<CUDAVisible*>* const visibles, Array<Material<CUDA_RNG>*>* const materials);

	__device__ CUDAScene(const CUDAScene& cs) = delete;
	__device__ CUDAScene(CUDAScene&& cs) = delete;
	__device__ CUDAScene& operator=(const CUDAScene& cs) = delete;
	__device__ CUDAScene& operator=(CUDAScene&& cs) = delete;

	__host__ ~CUDAScene();

	__host__ void set_visibles(Array<CUDAVisible*>* const new_visibles);
	__host__ void set_materials(Array<Material<CUDA_RNG>*>* const new_materials);

	Array<CUDAVisible*>* visibles;
	Array<Material<CUDA_RNG>*>* materials;
	Array<Array<vec3>*>* vertex_arrays;
	Array<Array<uint32_t>*>* index_arrays;

private:

	__host__ void delete_visibles();

	__host__ void delete_materials();

	__host__ void delete_vertex_arrays();

	__host__ void delete_index_arrays();

};

__global__ void cuda_delete_visibles(CUDAScene* scene);
