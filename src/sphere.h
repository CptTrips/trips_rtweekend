#ifndef SPHERE_H
#define SPHERE_H

#include "visible.h"
#include "material.h"

class Sphere: public Visible {
  public:
    Sphere();
    Sphere(vec3 O, float r, Material* m);
    ~Sphere();
    virtual std::unique_ptr<Intersection> intersect(const Ray& r, float tmin, float tmax) const;
    vec3 center;
    float radius;
    Material* material;
};

#endif
