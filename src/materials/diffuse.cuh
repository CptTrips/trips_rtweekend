#ifndef DIFFUSE_H
#define DIFFUSE_H

#include "material.h"

template<typename RNG_T>
class Diffuse : public Material<RNG_T>
{
  public:
	__host__ __device__ Diffuse(vec3 a) : Material<RNG_T>(a, 1.f, 0.f, 0.f, 0.f, 0.f) {}
    //__host__ __device__ virtual ~Diffuse() = default;
    //__host__ __device__ vec3 bounce(const vec3& r_in, const vec3& normal, RNG_T* const rng) const;
    //__host__ __device__ bool is_opaque() const {return true;}
};

//#include "diffuse.tu"
#endif
