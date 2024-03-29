#include "metal.h"

using namespace std;

Metal::Metal() {}

Metal::Metal(vec3 a, float r) : Material(a), roughness(r) {rng = RNG();}

Metal::~Metal() {}

void Metal::bounce(Ray const& r_in, Intersection& ixn, Ray& r_out) {

  vec3 scatter_dir = r_in.direction() - 2.*dot(r_in.direction(), ixn.normal)*ixn.normal;

  vec3 perturbed_scatter_dir;

  do {
    perturbed_scatter_dir = scatter_dir + roughness * rng.sample_uniform_sphere();
  } while (dot(perturbed_scatter_dir, ixn.normal) < 0);

  r_out = Ray(ixn.p, perturbed_scatter_dir);

}

bool Metal::is_opaque() { return true; }
