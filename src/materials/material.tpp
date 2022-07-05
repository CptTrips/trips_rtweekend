#include "material.h"

template<typename RNG_T>
Material<RNG_T>::Material()
	: albedo(vec3()), diffuse(0), metallic(0), dielectric(0)
{
}

template<typename RNG_T>
__host__ __device__ Material<RNG_T>::Material(const vec3& a, const float& diffuse, const float& metallic, const float& dielectric, const float& roughness, const float& refractive_index)
	: albedo(a), roughness(roughness), refractive_index(refractive_index)
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
	return true;//dielectric == 0;
}

template<typename RNG_T>
__host__ Material<RNG_T>* Material<RNG_T>::to_device() const
{
	Material<RNG_T>* device_material;

	checkCudaErrors(cudaMalloc(&device_material, sizeof(*this)));

	checkCudaErrors(cudaMemcpy(device_material, this, sizeof(*this), cudaMemcpyHostToDevice));

	return device_material;
}

template<typename RNG_T>
__host__ __device__ vec3 Material<RNG_T>::bounce(const vec3& r_in, const vec3& normal, RNG_T* const rng) const
{
	float x = rng->sample();

	if (x < diffuse)
		return bounce_diffuse(r_in, normal, rng);
	else if (x < diffuse + metallic)
		return bounce_metallic(r_in, normal, rng);
	else
		return bounce_dielectric(r_in, normal, rng);
}

template<typename RNG_T>
__host__ __device__ vec3 Material<RNG_T>::bounce_diffuse(const vec3& r_in, const vec3& normal, RNG_T* const rng) const
{

  vec3 scatter_dir = normal + 0.99f * rng->sample_uniform_sphere();

  return scatter_dir;
}

template<typename RNG_T>
__host__ __device__ vec3 Material<RNG_T>::bounce_metallic(const vec3 & r_in, const vec3& normal, RNG_T* const rng) const
{
	const vec3 scatter_dir = r_in - 2.f * dot(r_in, normal) * normal;

	const int inside = signbit(dot(scatter_dir, normal));

	vec3 perturbed_scatter_dir;

	do {
		perturbed_scatter_dir = scatter_dir + roughness * rng->sample_uniform_sphere();
	} while (signbit(dot(perturbed_scatter_dir, normal)) != inside);

	return  perturbed_scatter_dir;
}

template<typename RNG_T>
__host__ __device__ vec3 Material<RNG_T>::bounce_dielectric(const vec3 & k_in, const vec3& n, RNG_T* const rng) const
{
	vec3 k_out;

	vec3 unit_k_in = normalise(k_in);

	float cos_in = fmaxf(dot(unit_k_in, n), -1.f);

	// Assume normal points out of material
	bool into_material = (cos_in < 0.f);

	float index_ratio = into_material ? (1.f / refractive_index) : refractive_index;

	float sign = into_material ? -1.f : 1.f;


	// Calculate quantities we'll need to determine which case we're in
	vec3 k_in_normal = cos_in*n;

	vec3 k_in_tang = unit_k_in - k_in_normal;

	vec3 k_out_tang = index_ratio * k_in_tang;

	float norm2_k_out_tang = dot(k_out_tang, k_out_tang);

	vec3 k_reflected = unit_k_in - 2.f*k_in_normal;

	if (norm2_k_out_tang >= 1.f) { // Total internal reflection

		k_out = k_reflected;

	} else { // Refraction

		float r = reflectance(cos_in, index_ratio);

		if (r > rng->sample()) { // Stochastically sample reflected/refracted rays

			k_out = k_reflected;

		} else { // refract

			vec3 k_out_normal = sign * sqrt(1.f - norm2_k_out_tang) * n;

			k_out = k_out_normal + k_out_tang;

		}

	}

	return k_out;
}

template<typename RNG_T>
__host__ __device__ float Material<RNG_T>::reflectance(float cosine_in, float index_ratio) const
{

	float r0 = (1.f - index_ratio) / (1.f + index_ratio);

	r0 *= r0;

	return r0 + (1.f - r0)*powf(1.f + cosine_in, 5.f);
}



