
template<typename RNG_T>
__host__ __device__ vec3 Diffuse<RNG_T>::bounce(const vec3& r_in, const vec3& normal, RNG_T* const rng) const {

  vec3 scatter_dir = normal + 0.99frng->sample_uniform_sphere();

  return scatter_dir;
}

