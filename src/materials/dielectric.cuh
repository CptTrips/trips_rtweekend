#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "material.h"

template<typename RNG_T>
class Dielectric : public Material<RNG_T>
{
  public:
    __host__ __device__ Dielectric();
	__host__ __device__ Dielectric(vec3 a, float n) : Material<RNG_T>(a), refractive_index(n) {}
    //__host__ __device__ virtual ~Dielectric() = default;
    __host__ __device__ vec3 bounce(const vec3& r_in, const vec3& normal, RNG_T* const rng) const;
    __host__ __device__ float reflectance(const vec3& k_in, const vec3& k_tr, const vec3& n) const;
    __host__ __device__ bool is_opaque() const { return false; }

    float refractive_index;

  private:
    __host__ __device__ float reflectance_formula(float a, float b) const;
};

#include "dielectric.tu"
#endif
