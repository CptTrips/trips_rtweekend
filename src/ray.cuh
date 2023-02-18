#pragma once
#include <iostream>
#include "vec3.cuh"

class Ray
{
  public:
    __host__ __device__ Ray();
    __host__ __device__ Ray(const uint32_t& id, const vec3& o, const vec3& d);
    __host__ __device__ Ray(const uint32_t& id, const vec3& o, const vec3& d, const vec3& colour);
    __host__ __device__ vec3 origin() const;
    __host__ __device__ vec3 direction() const;
    __host__ __device__ vec3 point_at(float t) const;
    /*
    __host__ __device__ vec3& operator=(const vec3& rhs);
    __host__ __device__ vec3 operator=(vec3&& rhs);
    */

    uint64_t id;
    vec3 o;
    vec3 d;
    vec3 colour;
};
