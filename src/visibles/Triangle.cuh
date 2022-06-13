#pragma once
#include "CUDAVisible.cuh"
#include "../materials/material.h"
#include "../rand.h"

class Triangle : public CUDAVisible {
public:
    __host__ __device__ Triangle();
    //CUDASphere(const Sphere& s);
    __host__ __device__ Triangle(const vec3 points[3], Material<CUDA_RNG>* m);
    __host__ __device__ ~Triangle();
    __device__ virtual Intersection* intersect(const Ray& r, float tmin, float tmax) const;
    __device__ virtual Ray bounce(const vec3& r_in, const vec3& ixn_p, CUDA_RNG* rng) const;
    __device__ virtual vec3 albedo(const vec3& p) const;

    vec3 points[3];

    vec3 normal;

    Material<CUDA_RNG>* material;

private:

    __device__ bool lines_cross(const vec3& a0, const vec3& a1, const vec3& b0, const vec3& b1) const;
};
