
template<typename RNG_T>
__host__ __device__ vec3 Metal<RNG_T>::bounce(const vec3& r_in, const vec3& normal, RNG_T* const rng) const
{

  const vec3 scatter_dir = r_in - 2.f*dot(r_in, normal)*normal;

  const int inside = signbit(dot(scatter_dir, normal));

  vec3 perturbed_scatter_dir;

  do {
    perturbed_scatter_dir = scatter_dir + roughness * rng->sample_uniform_sphere();
  } while (signbit(dot(perturbed_scatter_dir, normal)) != inside);

  return  perturbed_scatter_dir;

}
