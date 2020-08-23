#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "rand.h"
#include "material.h"

class Dielectric : public Material
{
  public:
    Dielectric();
    Dielectric(vec3 albedo, float refractive_index);
    ~Dielectric();
    void bounce(Ray const& r_in, Intersection& ixn, Ray& r_out);
    float reflectance(const vec3& k_in, const vec3& k_tr, const vec3& n);
    bool is_opaque();

    float refractive_index;

  private:
    RNG rng;
    float reflectance_formula(float a, float b);
};

#endif
