#include "CUDASphere.cuh"

__host__ __device__ CUDASphere::CUDASphere()
{
    center = vec3(0.f, 0.f, 0.f);

    radius = 0.f;

    material = NULL;
}

/*
CUDASphere::CUDASphere(const Sphere& s)
{
    // Copy attributes

    const vec3* h_center_ptr = &(s.center);

    cudaMemcpy(&(this->center), h_center_ptr, sizeof(*h_center_ptr), cudaMemcpyHostToDevice);

    const float* h_radius_ptr = &(s.radius);

    cudaMemcpy(&(this->radius), h_radius_ptr, sizeof(float), cudaMemcpyHostToDevice);

    // Copy material
    Material* h_mat_ptr = s.material;
    size_t mat_size = sizeof(*(s.material));
    cudaMalloc(&this->material, mat_size);

    cudaMemcpy(this->material, h_mat_ptr, mat_size, cudaMemcpyHostToDevice);

    // Copy ptr to material
    cudaMemcpy(&(this->material), &h_mat_ptr, sizeof(h_mat_ptr), cudaMemcpyHostToDevice);

}
*/

__host__ __device__ CUDASphere::~CUDASphere() {
}

__device__ Intersection CUDASphere::intersect(const Ray& r, float tmin, float tmax) const {

    // a is 1 because ray directions should be normalised
    Intersection ixn;

    vec3 o_c = r.origin() - center;

    float b = dot(r.direction(), o_c);

    // discriminant / 4
    float disc_4 = b * b - (dot(o_c, o_c) - radius * radius);

    // (Real) Intersection solutions exist
    if (disc_4 > 0.f) {

        float sqrt_disc_4 = sqrt(disc_4);

        float t = -b - sqrt_disc_4;

        // Try to find earliest intersection
        if (t < tmax && t > tmin) {
            ixn = Intersection(t, normalise(r.point_at(t) - center), -1);
        }

        // We might be inside the sphere
        // (earliest intersection is behind ray origin, farthest is ahead of ray origin)
        /*
        else if (!material->is_opaque()) {

            t += 2.f * sqrt_disc_4;

            if (t < tmax && t > tmin) {
				ixn_ptr = new Intersection(t, -1);
            }
        }
        */
    }

    return ixn;
}

__device__ Ray CUDASphere::bounce(const vec3& r_in, const vec3& ixn_p, CUDA_RNG* rng) const
{
    vec3 normal = normalise(ixn_p - center);

    vec3 out_dir = material->scatter(r_in, normal, rng);

    return Ray(-1, ixn_p, out_dir);
}

__device__ vec3 CUDASphere::albedo(const vec3& p) const
{
    return vec3(1.f, 0.f, 1.f);
}
