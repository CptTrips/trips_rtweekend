#include "ray.h"

Ray::Ray() {}

Ray::Ray(const vec3& o_in, const vec3& d_in)
{
  o = o_in;
  d = normalise(d_in);
}

vec3 Ray::origin() const
{
  return o;
}

vec3 Ray::direction() const
{
  return d;
}

vec3 Ray::point_at(float t) const
{
  return o + t*d;
}
