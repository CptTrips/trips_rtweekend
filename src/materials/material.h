#pragma once

#include "../ray.cuh"
#include "../Intersection.h"
#include "../Error.cuh"

template<typename RNG_T>
class Material
{

    __host__ __device__ vec3 diffuse_scatter(const vec3 & r_in, const vec3& normal, RNG_T* const rng) const;
    __host__ __device__ vec3 metallic_scatter(const vec3 & r_in, const vec3& normal, RNG_T* const rng) const;
    __host__ __device__ vec3 dielectric_scatter(const vec3 & k_in, const vec3& n, RNG_T* const rng) const;
    __host__ __device__ float reflectance(float cosine_in, float index_ratio) const;

  public:
    __host__ __device__ Material();
    __host__ __device__ Material(const float& diffuse, const float& metallic, const float& dielectric, const float& roughness, const float& refractive_index);

    __host__ __device__ vec3 scatter(const vec3 & r_in, const vec3& normal, RNG_T* const rng) const;
    __host__ __device__ bool is_opaque() const;


    float diffuse;

    float metallic;
    
    float dielectric;

    float roughness;

    float refractive_index;
};

template<typename RNG_T>
Material<RNG_T>::Material()
	: diffuse(1), refractive_index(1)
{
}

template<typename RNG_T>
__host__ __device__ Material<RNG_T>::Material(const float& diffuse, const float& metallic, const float& dielectric, const float& roughness, const float& refractive_index)
	: roughness(roughness), refractive_index(refractive_index)
{
	if (diffuse < 0 || metallic < 0 || dielectric < 0)
		printf("Invalid material params");

	// Normalise behaviour probabilities

	const float total = diffuse + metallic + dielectric;

	this->diffuse = diffuse / total;

	this->metallic = metallic / total;

	this->dielectric = dielectric / total;
}


template<typename RNG_T>
__host__ __device__ bool Material<RNG_T>::is_opaque() const
{
	return dielectric == 0;
}


template<typename RNG_T>
__host__ __device__ vec3 Material<RNG_T>::scatter(const vec3& r_in, const vec3& normal, RNG_T* const rng) const
{
	float x = rng->sample();

	if (x <= diffuse)
		return diffuse_scatter(r_in, normal, rng);
	else if (x <= diffuse + metallic)
		return metallic_scatter(r_in, normal, rng);
	else
		return diffuse_scatter(r_in, normal, rng);//dielectric_scatter(r_in, normal, rng);
}

template<typename RNG_T>
__host__ __device__ vec3 Material<RNG_T>::diffuse_scatter(const vec3& r_in, const vec3& normal, RNG_T* const rng) const
{

  vec3 scatter_dir = normal + 0.99f * rng->sample_uniform_sphere();

  return normalise(scatter_dir);
}

template<typename RNG_T>
__host__ __device__ vec3 Material<RNG_T>::metallic_scatter(const vec3 & r_in, const vec3& normal, RNG_T* const rng) const
{

	vec3 specular, roughNormal;

	if (normal.length() == 0.f)
		printf("Bad normal\n");

	do
	{

		roughNormal = normalise(normal + (roughness * rng->sample_uniform_sphere()));

		specular = r_in - 2 * dot(roughNormal, r_in) * roughNormal;
	} while (dot(specular, normal) <= 0.f);

	return normalise(specular);
}

template<typename RNG_T>
__host__ __device__ vec3 Material<RNG_T>::dielectric_scatter(const vec3 & k_in, const vec3& n, RNG_T* const rng) const
{
	const vec3 unit_k_in = normalise(k_in);

	const vec3 unit_n = normalise(n);

	const float cos_in = fmaxf(dot(unit_k_in, unit_n), -1.f);

	// Assume normal points out of material
	const bool into_material = (cos_in < 0.f);

	const float index_ratio = into_material ? (1.f / refractive_index) : refractive_index;

	const vec3 out_normal = into_material ? -1.f * unit_n : unit_n;


	// Calculate quantities we'll need to determine which case we're in
	const vec3 k_in_normal = cos_in*unit_n;

	const vec3 k_in_tang = unit_k_in - k_in_normal;

	const vec3 k_out_tang = index_ratio * k_in_tang;

	const float norm2_k_out_tang = dot(k_out_tang, k_out_tang);

	const vec3 k_reflected = unit_k_in - 2.f*k_in_normal;

	const float r = reflectance(cos_in, index_ratio);

	vec3 k_out;

	if (norm2_k_out_tang >= 1.f) 
	{ // Total internal reflection

		k_out = k_reflected;

	}
	else if (r > rng->sample())
	{

		k_out = k_reflected;

	}
	else { // Refraction

		const vec3 k_out_normal = sqrt(1.f - norm2_k_out_tang) * out_normal;

		k_out = k_out_normal + k_out_tang;

	}

	return k_out;
}

template<typename RNG_T>
__host__ __device__ float Material<RNG_T>::reflectance(float cosine_in, float index_ratio) const
{

	float r0 = (1.f - index_ratio) / (1.f + index_ratio);

	r0 *= r0;

	return r0 + (1.f - r0)*powf(1.f - fabsf(cosine_in), 5.f);
}



