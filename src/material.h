#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "visible.h"

class Material
{
  public:
    Material();
    Material(vec3 a);
    virtual ~Material();
    virtual void bounce(Ray const& r_in, Intersection& ixn, Ray& r_out)=0;
    virtual bool is_opaque()=0;
    vec3 albedo;
};

#endif
