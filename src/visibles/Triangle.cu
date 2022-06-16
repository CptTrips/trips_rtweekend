#include "Triangle.cuh"

__host__ __device__ Triangle::Triangle()
{
	for (int i = 0; i < 3; i++)
		points[i] = vec3(0.f, 0.f, 0.f);

	material = NULL;
}

__host__ __device__ Triangle::Triangle(const vec3 points[3], const Material<CUDA_RNG>* m) : material(m)
{
	for (int i = 0; i < 3; i++)
		this->points[i] = points[i];

	normal = cross(points[1] - points[0], points[2] - points[1]);

	normal.normalise();
}

__host__ __device__ Triangle::Triangle(const Triangle& t)
{

	material = t.material;

	for (int i = 0; i < 3; i++)
	{
		points[i] = t.points[i];
	}

	normal = t.normal;

}

__host__ __device__ Triangle& Triangle::operator=(const Triangle& t)
{

	if (this == &t)
		return *this;

	material = t.material;

	for (int i = 0; i < 3; i++)
	{
		points[i] = t.points[i];
	}

	normal = t.normal;

	return *this;

}

__host__ __device__ Triangle::~Triangle()
{
}

__device__ Intersection* Triangle::intersect(const Ray& r, float tmin, float tmax) const
{

	Intersection* ixn = NULL;

	// Do not scatter from the back side if the material is opaque
	if (material->is_opaque() && dot(r.d, normal) > 0.f)
		return NULL;

	// Find the point the ray intersects triangle's plane
	const float t = dot((points[0] - r.o), normal) / dot(r.d, normal);

	if (t < tmin || t > tmax)
		return NULL;

	// Determine if this point lies inside the triangle
	const vec3 p = r.point_at(t);

	if (point_inside(p))
	{
		ixn = new Intersection(t, this);
	}

	return ixn;
}

__device__ bool Triangle::point_inside(const vec3& p) const
{
	const vec3 outside_point = points[0] - (points[1] - points[0]) - (points[2] - points[0]);

	int crosses = 0;

	for (int i = 0; i < 3; i++)
	{
		int j = (i + 1) % 3;
		if (this->lines_cross(p, outside_point, points[i], points[j]))
			crosses++;
	}

	return (crosses % 2 == 1);
}

__device__ bool Triangle::lines_cross(const vec3& a0, const vec3& a1, const vec3& b0, const vec3& b1) const
{

	const vec3 d = b0 - a0;

	const vec3 la = a1 - a0;

	const vec3 lb = b1 - b0;

	const float s = dot(cross(d, la), normal) / dot(cross(la, lb), normal);

	const float r = (dot(d, la) + s * dot(lb, la)) / dot(la, la);

	return (0 < r && r <= 1 && 0 < s && s <= 1);
}

__device__ Ray Triangle::bounce(const vec3& r_in, const vec3& ixn_p, CUDA_RNG* rng) const
{
	const vec3 r_out = material->bounce(r_in, normal, rng);

	return Ray(ixn_p, r_out);
}

__device__ vec3 Triangle::albedo(const vec3& p) const
{

	return material->albedo;
}

