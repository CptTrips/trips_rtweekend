#ifndef RAND_H
#define RAND_H

#include "vec3.h"
#include <random>

class RNG {
  public:
    RNG();
    float sample();
    vec3 sample_uniform_sphere();

};

#endif
