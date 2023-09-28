#include "sphere.cuh"

Sphere::Sphere() {}

Sphere::Sphere(vec3 O, float r, Material<CPU_RNG>* m) : center(O), radius(r), material(m) {}

CUDAVisible* Sphere::to_device() const
{
    // Allocate CUDASphere on device
    CUDASphere* d_sphere_ptr;

    size_t s_size = sizeof(*this);

    cudaMalloc(&d_sphere_ptr, s_size);

    // Copy attributes

    vec3* d_center_ptr = &(d_sphere_ptr->center);

    cudaMemcpy(d_center_ptr, &(this->center), sizeof(*d_center_ptr), cudaMemcpyHostToDevice);

    float* d_radius_ptr = &(d_sphere_ptr->radius);

    cudaMemcpy(d_radius_ptr, &(this->radius), sizeof(float), cudaMemcpyHostToDevice);

    // Copy material
    Material<CPU_RNG>* d_mat_ptr;
    size_t mat_size = sizeof(*(this->material));
    cudaMalloc(&d_mat_ptr, mat_size);

    cudaMemcpy(d_mat_ptr, this->material, mat_size, cudaMemcpyHostToDevice);

    // Copy ptr to material
    cudaMemcpy(&(d_sphere_ptr->material), &(this->material), sizeof(d_mat_ptr), cudaMemcpyHostToDevice);

    return d_sphere_ptr;
}

Sphere::~Sphere()
{
  delete material;
}

std::unique_ptr<Intersection> Sphere::intersect(const Ray& r, float tmin, float tmax) const{

  std::unique_ptr<Intersection> ixn_ptr{};

  // a is 1 because ray directions should be normalised

  vec3 o_c = r.origin() - center;

  float b = dot(r.direction(), o_c);

  // discriminant / 4
  float disc_4 = b*b - (dot(o_c, o_c) - radius*radius);

  if (disc_4 > 0) {

    float sqrt_disc_4 = sqrt(disc_4);

    float t = -b - sqrt_disc_4;

    // Try to find earliest intersection
    if (t < tmax && t > tmin) {
	  ixn_ptr = std::unique_ptr<Intersection>(new Intersection(t, this));
    }

    // We might be inside the sphere
    // (earliest intersection is behind ray origin, farthest is ahead of ray origin)
    else if (!material->is_opaque()) {

      t += 2.*sqrt_disc_4;

      if (t < tmax && t > tmin) {
		ixn_ptr = std::unique_ptr<Intersection>(new Intersection(t, this));
      }
    }
  }

  return ixn_ptr;
}
