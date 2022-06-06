#include "ray.cuh"

__host__ __device__ Ray::Ray() {}

__host__ __device__ Ray::Ray(const vec3& o_in, const vec3& d_in)
{
  o = o_in;
  d = normalise(d_in);
}

__host__ __device__ vec3 Ray::origin() const
{
  return o;
}

__host__ __device__ vec3 Ray::direction() const
{
  return d;
}

__host__ __device__ vec3 Ray::point_at(float t) const
{
  return o + t*d;
}
