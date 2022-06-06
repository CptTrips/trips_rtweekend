#ifndef DIFFUSE_H
#define DIFFUSE_H

#include "../rand.h"
#include "material.h"

class Diffuse : public Material
{
  public:
    Diffuse();
    Diffuse(vec3 albedo);
    ~Diffuse();
    void bounce(Ray const& r_in, Intersection& ixn, Ray& r_out) const;
    bool is_opaque() const;

  private:
    RNG rng;
};
#endif
