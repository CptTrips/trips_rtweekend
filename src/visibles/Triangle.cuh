#pragma once
#include "CUDAVisible.cuh"
#include "../materials/material.h"
#include "../rand.h"
#include "../Array.cuh"

class Triangle : public CUDAVisible {
public:
    __host__ __device__ Triangle();
    __host__ __device__ Triangle(const vec3* points, const Material<CUDA_RNG>* m);
    __host__ __device__ Triangle(const vec3& a, const vec3& b, const vec3& c);

    __host__ __device__ Triangle(const Triangle& t);
    __host__ __device__ Triangle(Triangle&& t);
    __host__ __device__ Triangle& operator=(Triangle t);

    __host__ __device__ ~Triangle();

    __device__ virtual Intersection intersect(const Ray& r, float tmin, float tmax) const;
    __device__ virtual Ray bounce(const vec3& r_in, const vec3& ixn_p, CUDA_RNG* rng) const;
    __device__ virtual vec3 albedo(const vec3& p) const;

    Array<vec3>* points;

    vec3 normal;

    const Material<CUDA_RNG>* material;

private:

    __host__ __device__ friend void swap(Triangle& a, Triangle& b);

    __device__ bool lines_cross(const vec3& a0, const vec3& a1, const vec3& b0, const vec3& b1) const;
    __device__ bool point_inside(const vec3& p) const;
};
