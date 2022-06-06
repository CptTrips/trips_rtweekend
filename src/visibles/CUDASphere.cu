#include "CUDASphere.cuh"

CUDASphere::CUDASphere() {}

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

CUDASphere::~CUDASphere() {
    delete material;
}

__device__ Intersection* CUDASphere::intersect(const Ray& r, float tmin, float tmax) const {

    // a is 1 because ray directions should be normalised

    Intersection* ixn_ptr = NULL;

    vec3 o_c = r.origin() - center;

    float b = dot(r.direction(), o_c);

    // discriminant / 4
    float disc_4 = b * b - (dot(o_c, o_c) - radius * radius);

    if (disc_4 > 0) {

        float sqrt_disc_4 = sqrt(disc_4);

        float t = -b - sqrt_disc_4;

        // Try to find earliest intersection
        if (t < tmax && t > tmin) {
            vec3 ixn_point = r.point_at(t);
            ixn_ptr = new Intersection(t, ixn_point, (ixn_point - center) / radius, material);
        }

        // We might be inside the sphere
        // (earliest intersection is behind ray origin, farthest is ahead of ray origin)
        else if (!material->is_opaque()) {

            t += 2. * sqrt_disc_4;

            if (t < tmax && t > tmin) {
                vec3 ixn_point = r.point_at(t);
                ixn_ptr = new Intersection( t, ixn_point, (ixn_point - center) / radius, material );
            }
        }
    }

    return ixn_ptr;
}
