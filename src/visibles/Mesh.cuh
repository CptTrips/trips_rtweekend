
#pragma once
#include "CUDAVisible.cuh"
#include "../rand.h"
#include "../Array.cuh"
#include "../materials/material.h"
#include "TriangleView.cuh"
#include "Triangle.cuh"
#include <assert.h>

class Mesh : public CUDAVisible {
public:
    __host__ __device__ Mesh(const Array<vec3>* const vertices, const Array<uint32_t>* const indices, const Material<CUDA_RNG>* const material);
    __host__ __device__ ~Mesh();
    __device__ virtual Intersection* intersect(const Ray& r, float tmin, float tmax) const;
    __device__ virtual Ray bounce(const vec3& r_in, const vec3& ixn_p, CUDA_RNG* rng) const;
    __device__ virtual vec3 albedo(const vec3& p) const;

private:
    __device__ bool intersect_bbox(const Ray& r, float tmin, float tmax) const;
    __host__ __device__ void construct_bbox(const vec3& min, const vec3& max);

    Array<Triangle> bbox_triangles;
    const Array<vec3>* const vertices;
    const Array<uint32_t>* const indices;
    Array<TriangleView> triangles;

    const Material<CUDA_RNG>* const material;

};
