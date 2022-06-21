#pragma once

#include <cuda_runtime.h>
#include "materials\material.h"
#include "visibles\CUDAVisible.cuh"
#include "rand.h"
#include "Array.cuh"
#include "Managed.cuh"
#include "Error.cuh"


class CUDAScene : public Managed
{
public:
	__host__ __device__ CUDAScene();
	__device__ CUDAScene(Array<CUDAVisible*>* const visibles, Array<const Material<CUDA_RNG>*>* const materials);

	__device__ CUDAScene(const CUDAScene& cs);
	__device__ CUDAScene(CUDAScene&& cs);
	__device__ CUDAScene& operator=(const CUDAScene& cs);
	__device__ CUDAScene& operator=(CUDAScene&& cs);

	__host__ __device__ ~CUDAScene();

	__device__ CUDAVisible* operator[](const uint32_t i);
	__device__ const CUDAVisible* operator[](const uint32_t i) const;

	__device__ void set_visibles(Array<CUDAVisible*>* const new_visibles);
	__device__ void set_materials(Array<const Material<CUDA_RNG>*>* const new_materials);

	__device__ uint32_t size() const;

	Array<CUDAVisible*>* visibles;
	Array<const Material<CUDA_RNG>*>* materials;
	Array<const Array<vec3>*>* vertex_arrays;
	Array<const Array<uint32_t>*>* index_arrays;

private:

	__host__ __device__ void delete_visibles();
	__host__ __device__ void delete_materials();

};
