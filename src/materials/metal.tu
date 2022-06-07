
template<typename RNG_T>
__host__ __device__ vec3 Metal<RNG_T>::bounce(const vec3& r_in, const vec3& normal, RNG_T* const rng) const
{

  vec3 scatter_dir = r_in - 2.*dot(r_in, normal)*normal;

  vec3 perturbed_scatter_dir;

  do {
    perturbed_scatter_dir = scatter_dir + roughness * rng->sample_uniform_sphere();
  } while (dot(perturbed_scatter_dir, normal) < 0);

  return  perturbed_scatter_dir;

}
