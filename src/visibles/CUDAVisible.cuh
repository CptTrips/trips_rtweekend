#pragma once
#include <memory>
#include "../ray.cuh"
#include "../Intersection.h"

class CUDAVisible {
public:
    __device__ virtual Intersection* intersect(const Ray& r, float t_min, float t_max) const = 0;
    size_t size() { return sizeof(*this); }
};

