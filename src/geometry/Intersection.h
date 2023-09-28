#pragma once
#include "maths/vec3.cuh"

struct Intersection {
  float t;
  vec3 normal;
  uint32_t id;
    __host__ __device__ Intersection(const float t, const vec3& normal, const uint32_t& id) : t(t), normal(normal), id(id) {};
    __host__ __device__ Intersection() : t(INFINITY), normal(), id(-1) {};
};
