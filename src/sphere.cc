#include "sphere.h"

Sphere::Sphere() {}

Sphere::Sphere(vec3 O, float r, Material* m) : center(O), radius(r), material(m) {}

Sphere::~Sphere() {
  delete material;
}

bool Sphere::intersect(const Ray& r, float tmin, float tmax, Intersection& ixn) const{

  // a is 1 because ray directions should be normalised

  vec3 o_c = r.origin() - center;

  float b = dot(r.direction(), o_c);

  // discriminant / 4
  float disc_4 = b*b - (dot(o_c, o_c) - radius*radius);

  if (disc_4 > 0) {

    float sqrt_disc_4 = sqrt(disc_4);

    float t = -b - sqrt_disc_4;

    // Try to find earliest intersection
    if (t < tmax && t > tmin) {
      ixn.t = t;
      ixn.p = r.point_at(t);
      ixn.normal = (ixn.p - center) / radius;
      ixn.material = material;
      return true;
    }

    // We might be inside the sphere
    // (earliest intersection is behind ray origin, farthest is ahead of ray origin)
    if (!material->is_opaque()) {

      t += 2.*sqrt_disc_4;

      if (t < tmax && t > tmin) {
        ixn.t = t;
        ixn.p = r.point_at(t);
        ixn.normal = (ixn.p - center) / radius;
        ixn.material = material;
        return true;
      }
    }
  }

  return false;
}
