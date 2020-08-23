#ifndef METAL_H
#define METAL_H

#include "rand.h"
#include "material.h"

class Metal : public Material
{
  public:
    Metal();
    Metal(vec3 albedo, float roughness);
    ~Metal();
    void bounce(Ray const& r_in, Intersection& ixn, Ray& r_out);
    bool is_opaque();
    float roughness;

  private:
    RNG rng;
};
#endif
