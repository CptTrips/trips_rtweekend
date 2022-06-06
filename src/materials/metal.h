#ifndef METAL_H
#define METAL_H

#include "../rand.h"
#include "material.h"

class Metal : public Material
{
  public:
    Metal();
    Metal(vec3 albedo, float roughness);
    ~Metal();
    __host__ __device__ void bounce(Ray const& r_in, Intersection& ixn, Ray& r_out) const;
    __host__ __device__ bool is_opaque() const;
    float roughness;

  private:
    RNG rng;
};
#endif
