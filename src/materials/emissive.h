#ifndef EMISSIVE_H
#define EMISSIVE_H

class Emissive: public Visible {
  public:
    virtual bool intersect(const Ray& r, float t_min, float t_max, Intersection& ixn);
    vec3 color;
    float intensity;
};

#endif
