#ifndef MATERIAL_H
#define MATERIAL_H

#include "../ray.cuh"
#include "../Intersection.h"
#include "../Error.cuh"

template<typename RNG_T>
class Material
{

    __host__ __device__ vec3 bounce_diffuse(const vec3 & r_in, const vec3& normal, RNG_T* const rng) const;
    __host__ __device__ vec3 bounce_metallic(const vec3 & r_in, const vec3& normal, RNG_T* const rng) const;
    __host__ __device__ vec3 bounce_dielectric(const vec3 & k_in, const vec3& n, RNG_T* const rng) const;
    __host__ __device__ float reflectance(float cosine_in, float index_ratio) const;

  public:
    __host__ __device__ Material();
    __host__ __device__ Material(const vec3& a, const float& diffuse, const float& metallic, const float& dielectric, const float& roughness, const float& refractive_index);

    __host__ __device__ vec3 bounce(const vec3 & r_in, const vec3& normal, RNG_T* const rng) const;
    __host__ __device__ bool is_opaque() const;

    __host__ Material<RNG_T>* to_device() const;


    vec3 albedo;

    float diffuse;

    float metallic;
    
    float dielectric;

    float roughness;

    float refractive_index;
};

#include "material.tpp"

#endif
