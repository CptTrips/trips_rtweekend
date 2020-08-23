#ifndef VISIBLE_H
#define VISIBLE_H

#include "ray.h"

class Material;

struct Intersection {
  float t;
  vec3 p;
  vec3 normal;
  Material* material;
};

class Visible {
  public:
    virtual bool intersect(const Ray& r, float t_min, float t_max, Intersection& ixn) const = 0;
    virtual ~Visible() {};
};

#endif
