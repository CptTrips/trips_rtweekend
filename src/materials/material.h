#ifndef MATERIAL_H
#define MATERIAL_H

#include "../ray.cuh"
#include "../Intersection.h"

class Material
{
  public:
    Material();
    Material(vec3 a);
    virtual ~Material();
    __host__ __device__ virtual void bounce(Ray const& r_in, Intersection& ixn, Ray& r_out) const = 0;
    __host__ __device__ virtual bool is_opaque() const = 0;
    vec3 albedo;
};

#endif
