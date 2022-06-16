

template<typename RNG_T>
__host__ __device__ vec3 Dielectric<RNG_T>::bounce(const vec3& r_in, const vec3& normal, RNG_T* const rng) const{

  vec3 k_out;

  vec3 k_in = r_in;

  vec3 n = normal;

  // Figure out the index of the incoming/outcoming ray
  float in_index, out_index;
  float sign;

  if (dot(k_in, n) < 0.) {
    in_index = 1.;
    out_index = refractive_index;
    sign = -1.;
  } else {
    in_index = refractive_index;
    out_index = 1.;
    sign = 1.;
  }

  // Calculate quantities we'll need to determine which case we're in
  vec3 k_in_normal = dot(k_in, n)*n;

  vec3 k_in_tang = k_in - k_in_normal;

  vec3 k_out_tang = (in_index / out_index) * k_in_tang;

  float norm2_k_out_tang = dot(k_out_tang, k_out_tang);

  vec3 k_reflected = k_in - 2.*k_in_normal;

  vec3 k_refracted = sign*sqrt(1. - norm2_k_out_tang) * n + k_out_tang;

  if (norm2_k_out_tang > 1.) { // Total internal reflection
    k_out = k_reflected;
  } else { //Refraction
    float r = reflectance(r_in, k_refracted, normal);

    if (rng->sample() > r) { // Stochastically sample reflected/refracted rays
      k_out = k_refracted;
    } else { // reflect
      k_out = k_reflected;
    }
  }

  r_out = k_out;
}

template<typename RNG_T>
__host__ __device__ float Dielectric<RNG_T>::reflectance(const vec3& k_in, const vec3& k_out, const vec3& n) const{

  float n_in, n_out;

  float cos_i = dot(k_in, n);
  float cos_o = dot(k_out, n);

  if (cos_i < 0.) {
    n_in = 1.;
    n_out = refractive_index;
  } else {
    n_in = refractive_index;
    n_out = 1.;
  }

  float r_s = reflectance_formula(n_in * cos_i, n_out * cos_o);

  float r_p = reflectance_formula(n_in * cos_o, n_out * cos_i);

  return 0.5f * (r_s + r_p);
}

template<typename RNG_T>
__host__ __device__ float Dielectric<RNG_T>::reflectance_formula(float a, float b) const{

  float c = (a - b)/(a + b);

  return c*c;
}