#ifndef MATERIAL_H
#define MATERIAL_H

#include "../ray.cuh"
#include "../Intersection.h"

template<typename RNG_T>
class Material
{
  public:
    __host__ __device__ Material();
    __host__ __device__ Material(vec3 a) : albedo(a) {}

    //__host__ __device__ virtual ~Material();

    __host__ __device__ virtual vec3 bounce(const vec3 & r_in, const vec3& normal, RNG_T* const rng) const = 0;
    __host__ __device__ virtual bool is_opaque() const = 0;
    vec3 albedo;

};

#endif
