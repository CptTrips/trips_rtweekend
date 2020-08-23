#include "diffuse.h"

Diffuse::Diffuse() {}

Diffuse::Diffuse(vec3 a) : Material(a), rng(RNG()) {}

Diffuse::~Diffuse() {}

void Diffuse::bounce(Ray const& r_in, Intersection& ixn, Ray& r_out) {

  vec3 scatter_dir = ixn.normal + rng.sample_uniform_sphere();

  r_out = Ray(ixn.p, scatter_dir);
}

bool Diffuse::is_opaque() { return true; }
