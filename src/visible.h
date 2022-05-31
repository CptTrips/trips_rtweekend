#ifndef VISIBLE_H
#define VISIBLE_H

#include <memory>
#include "ray.h"

class Material;

struct Intersection {
  float t;
  vec3 p;
  vec3 normal;
  Material* const material;
};

class Visible {
  public:
    virtual std::unique_ptr<Intersection> intersect(const Ray& r, float t_min, float t_max) const = 0;
    virtual ~Visible() {};
};

#endif
