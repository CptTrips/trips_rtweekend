#pragma once
#include "vec3.cuh"

class CUDAVisible;

class Intersection {
public:
  const float t;
  const CUDAVisible* const visible;
    __host__ __device__ Intersection(const float t, const CUDAVisible* const visible) : t(t), visible(visible) {};
    __host__ __device__ Intersection() : t(0), visible(NULL) {};
};
