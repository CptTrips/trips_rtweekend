#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <fstream>
#include <json.hpp>
#include <string>

#include "materials\material.h"
#include "visibles\CUDAVisible.cuh"
#include "visibles\CUDASphere.cuh"
#include "rand.h"
#include "Array.cuh"
#include "Managed.cuh"
#include "Error.cuh"
#include "UnifiedArray.cuh"


class CUDAScene : public Managed
{
public:
	__host__ CUDAScene();
	__host__ CUDAScene(UnifiedArray<CUDAVisible*>* const visibles, UnifiedArray<Material<CUDA_RNG>*>* const materials);
	__host__ CUDAScene(const std::string& fp);
	__host__ CUDAScene(const unsigned int visible_count, const unsigned int material_count);

	__device__ CUDAScene(const CUDAScene& cs) = delete;
	__device__ CUDAScene(CUDAScene&& cs) = delete;
	__device__ CUDAScene& operator=(const CUDAScene& cs) = delete;
	__device__ CUDAScene& operator=(CUDAScene&& cs) = delete;

	__host__ ~CUDAScene();

	__host__ void set_visibles(UnifiedArray<CUDAVisible*>* const new_visibles);
	__host__ void set_materials(UnifiedArray<Material<CUDA_RNG>*>* const new_materials);

	UnifiedArray<CUDAVisible*>* visibles;
	UnifiedArray<Material<CUDA_RNG>*>* materials;
	UnifiedArray<Array<vec3>*>* vertex_arrays;
	UnifiedArray<Array<uint32_t>*>* index_arrays;

private:

	__host__ void delete_visibles();

	__host__ void delete_materials();

	__host__ void delete_vertex_arrays();

	__host__ void delete_index_arrays();

};

__global__ void instantiate_spheres(CUDAScene* const scene, const UnifiedArray<CUDASphere>* const spheres);

__global__ void cuda_delete_visibles(UnifiedArray<CUDAVisible*>* visibles);
