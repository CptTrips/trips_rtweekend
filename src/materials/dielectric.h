#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "../rand.h"
#include "material.h"

class Dielectric : public Material
{
  public:
    Dielectric();
    Dielectric(vec3 albedo, float refractive_index);
    ~Dielectric();
    __host__ __device__ void bounce(Ray const& r_in, Intersection& ixn, Ray& r_out) const;
    __host__ __device__ float reflectance(const vec3& k_in, const vec3& k_tr, const vec3& n) const;
    __host__ __device__ bool is_opaque() const;

    float refractive_index;

  private:
    RNG rng;
    __host__ __device__ float reflectance_formula(float a, float b) const;
};

#endif
