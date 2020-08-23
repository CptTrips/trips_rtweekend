#ifndef DIFFUSE_H
#define DIFFUSE_H

#include "rand.h"
#include "material.h"

class Diffuse : public Material
{
  public:
    Diffuse();
    Diffuse(vec3 albedo);
    ~Diffuse();
    void bounce(Ray const& r_in, Intersection& ixn, Ray& r_out);
    bool is_opaque();

  private:
    RNG rng;
};
#endif
