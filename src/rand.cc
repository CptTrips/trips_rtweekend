#include "rand.h"

std::mt19937 gen;

std::uniform_real_distribution<float> rnd(0., 1.);

RNG::RNG() {}

float RNG::sample() const {

  return rnd(gen);
}

// Return a point sampled uniformly from sphere of radius 1
vec3 RNG::sample_uniform_sphere() const {

  vec3 p;

  do {
    p = 2.*vec3(sample()-.5, sample()-.5, sample()-.5);
  } while (dot(p,p) > 1.);

  return p;
}

