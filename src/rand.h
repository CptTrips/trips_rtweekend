#ifndef RAND_H
#define RAND_H

#include "vec3.h"
#include <random>

class RNG {
  public:
    RNG();
    float sample() const;
    vec3 sample_uniform_sphere() const;

};

#endif
