#ifndef METAL_H
#define METAL_H

#include "material.h"

template<typename RNG_T>
class Metal : public Material<RNG_T>
{
  public:
	__host__ __device__ Metal(vec3 albedo, float roughness) : Material<RNG_T>(albedo), roughness(roughness) {}
    //__host__ __device__ virtual ~Metal() = default;
    __host__ __device__ vec3 bounce(const vec3& r_in, const vec3& normal, RNG_T* const rng) const;
    __host__ __device__ bool is_opaque() const { return true; }
    float roughness;
};

#include "metal.tu"
#endif
