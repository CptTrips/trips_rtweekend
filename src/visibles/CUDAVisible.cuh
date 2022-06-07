#pragma once
#include <memory>
#include "../ray.cuh"
#include "../Intersection.h"
#include "../rand.h"

class CUDAVisible {
public:
    __device__ virtual Intersection* intersect(const Ray& r, float t_min, float t_max) const = 0;
    __device__ virtual Ray bounce(const vec3& r_in, const vec3& ixn_p, CUDA_RNG* rng) const = 0;
    __device__ virtual vec3 albedo(const vec3& p) const = 0;
    size_t size() { return sizeof(*this); }
};

