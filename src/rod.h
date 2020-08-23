#ifndef ROD_H
#define ROD_H

#include "visible.h"
#include "material.h"

class Rod: public Visible {
  public:
    Rod();
    Rod(vec3 c, float r, float l, Material* m);
    ~Rod();
    bool intersect(const Ray& r, float tmin, float tmax, Intersection& ixn) const;
    vec3 center;
    float radius;
    float length_;
    Material* material;

#endif
