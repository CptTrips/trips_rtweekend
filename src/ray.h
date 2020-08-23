#ifndef RAY_H
#define RAY_H

#include <iostream>
#include "vec3.h"

class Ray
{
  public:
    Ray();
    Ray(const vec3& o, const vec3& d);
    vec3 origin() const;
    vec3 direction() const;
    vec3 point_at(float t) const;

    vec3 o;
    vec3 d;

};

#endif
