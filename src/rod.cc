#include "rod.h"

Rod::Rod() {}

Rod::Rod(vec3 e, float r, vec3 l, Material* m) : end(e), radius(r),
  length_(l), material(m) {}

Rod::~Rod() {
  delete material;
}

bool Rod::intersect(const Ray& r, float tmin, float tmax, Intersection& ixn) const
{

  // Find ray's point of closest approach to 
