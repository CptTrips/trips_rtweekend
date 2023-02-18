#include "ray.cuh"

__host__ __device__ Ray::Ray() {}

__host__ __device__ Ray::Ray(const uint32_t& id, const vec3& o_in, const vec3& d_in) : id(id), colour(1.0f, 1.0f, 1.0f)
{

  o = o_in;
  d = normalise(d_in);
}

Ray::Ray(const uint32_t& id, const vec3& o, const vec3& d, const vec3& colour) : id(id), o(o), d(d), colour(colour)
{
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
