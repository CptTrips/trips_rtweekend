#pragma once
#include "vec3.cuh"

class Material;

class Intersection {
public:
  const float t;
  const vec3 p;
  const vec3 normal;
  const Material* const material;
    __host__ __device__ Intersection(const float t, const vec3& p, const vec3& normal, const Material* const material) : t(t), p(p), normal(normal), material(material) {};
    __host__ __device__ Intersection() : t(0), p(0, 0, 0), normal(0, 0, 0), material(NULL) {};
};
