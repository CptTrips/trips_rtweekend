#pragma once

#include "CUDAVisible.cuh"
#include "../rand.h"
#include "../Array.cuh"
#include "../materials/material.h"
#include "Triangle.cuh"

class TriangleView : public CUDAVisible
{
public:
	__host__ __device__ TriangleView();
	__host__ __device__ TriangleView(const Array<vec3>* const vertex_array, const Array<uint32_t>* const index_array, const uint32_t& index_0, const Material<CUDA_RNG>* const material);

	__host__ __device__ TriangleView(const TriangleView& tv);
	__host__ __device__ TriangleView& operator=(const TriangleView& tv);

	__host__ __device__ ~TriangleView();

    __device__ virtual Intersection* intersect(const Ray& r, float tmin, float tmax) const;
    __device__ virtual Ray bounce(const vec3& r_in, const vec3& ixn_p, CUDA_RNG* rng) const;
    __device__ virtual vec3 albedo(const vec3& p) const;

private:

	__device__ Triangle construct_triangle() const;


	const Array<vec3>* vertex_array;

	const Array<uint32_t>* index_array;

	uint32_t index_array_offset;

	const Material<CUDA_RNG>* material;

};