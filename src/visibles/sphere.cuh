#ifndef SPHERE_H
#define SPHERE_H

#include "CUDASphere.cuh"
#include "CUDAVisible.cuh"
#include "visible.h"
#include "../materials/material.h"
#include "../rand.h"

class Sphere: public Visible {
  public:
    Sphere();
    Sphere(vec3 O, float r, Material<CPU_RNG>* m);
    ~Sphere();
    virtual std::unique_ptr<Intersection> intersect(const Ray& r, float tmin, float tmax) const;
    virtual CUDAVisible* to_device() const;
    vec3 center;
    float radius;
    Material<CPU_RNG>* material;
};

#endif
