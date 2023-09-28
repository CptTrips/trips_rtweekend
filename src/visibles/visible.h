#ifndef VISIBLE_H
#define VISIBLE_H

#include <memory>
#include "geometry/ray.cuh"
#include "CUDAVisible.cuh"




class Visible {
  public:
    virtual std::unique_ptr<Intersection> intersect(const Ray& r, float t_min, float t_max) const = 0;
    size_t size() { return sizeof(*this); }
    virtual CUDAVisible* to_device() const = 0;
};



#endif
