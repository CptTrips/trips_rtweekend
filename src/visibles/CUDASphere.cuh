#pragma once
#include "CUDAVisible.cuh"
//#include "sphere.h"
#include "../materials/material.h"

class CUDASphere : public CUDAVisible {
public:
    CUDASphere();
    //CUDASphere(const Sphere& s);
    ~CUDASphere();
    __device__ virtual Intersection* intersect(const Ray& r, float tmin, float tmax) const;
    vec3 center;
    float radius;
    Material* material;
};
