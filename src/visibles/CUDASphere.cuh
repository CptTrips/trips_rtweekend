#pragma once
#include "CUDAVisible.cuh"
#include "../materials/material.h"
#include "../rand.h"

class CUDASphere : public CUDAVisible {
public:
    __host__ __device__ CUDASphere();
    //CUDASphere(const Sphere& s);
	__host__ __device__ CUDASphere(vec3 O, float r, Material<CUDA_RNG>* m) : center(O), radius(r), material(m) {}
    __host__ __device__ ~CUDASphere();
    __device__ virtual Intersection* intersect(const Ray& r, float tmin, float tmax) const;
    __device__ virtual Ray bounce(const vec3& r_in, const vec3& ixn_p, CUDA_RNG* rng) const;
    __device__ virtual vec3 albedo(const vec3& p) const;
    vec3 center;
    float radius;
    Material<CUDA_RNG>* material;
};
