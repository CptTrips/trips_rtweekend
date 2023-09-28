#include "Triangle.cuh"

__host__ __device__ Triangle::Triangle()
{
}

__host__ __device__ Triangle::Triangle(const vec3* points, const Material<CUDA_RNG>* m)
	: Triangle(points[0], points[1], points[2])
{
	material = m;
}

Triangle::Triangle(const vec3& a, const vec3& b, const vec3& c)
	: a(a), b(b), c(c), normal(cross(b - a, c - b).normalise())
{
}


Triangle::Triangle(const vec3& a, const vec3& b, const vec3& c, const vec3& n)
	: a(a), b(b), c(c), normal(n)
{
}

/*
__host__ __device__ Triangle::Triangle(const Triangle& t)
	: material(t.material),
	points(NULL),
	normal(t.normal)
{

	if (t.points)
	{
		points = new Array<vec3>(3);

		// Copy assign points
		*points = *t.points;
	}

}

Triangle::Triangle(Triangle&& t)
	: Triangle()
{
	swap(*this, t);
}


__host__ __device__ Triangle& Triangle::operator=(Triangle t)
{

	// t is either copy or move-constructed, as determined by the compiler
	// Using swap ensures that t is in a state to be deleted or assigned (i.e. null-constructed)
	swap(*this, t);

	return *this;

}

__host__ __device__ Triangle::~Triangle()
{
	if (points)
		delete points;
}
*/

__device__ Intersection Triangle::intersect(const Ray& r, float tmin, float tmax) const
{

	Intersection ixn;

	/*
	const Material<CUDA_RNG>* local_mat_ptr = material;

	Material<CUDA_RNG> local_material = *local_mat_ptr;
	*/

	// Do not scatter from the back side if the material is opaque
	/*
	if (material->is_opaque() && dot(r.d, normal) > 0.f)
		return NULL;
	*/

	if (dot(r.d, normal) > 0.f)
		return ixn;

	// Find the point the ray intersects triangle's plane
	const float t = dot((a - r.o), normal) / dot(r.d, normal);

	if (t < tmin || t > tmax)
		return ixn;

	// Determine if this point lies inside the triangle
	const vec3 p = r.point_at(t);

	if (point_inside(p))
	{
		ixn = Intersection(t, normal, -1);
	}

	return ixn;
}

__device__ bool Triangle::point_inside(const vec3& p) const
{
	const vec3 outside_point = a - (b - a) - (c - a);

	int crosses = 0;

	const vec3 vertices[] { a, b, c };

	for (int i = 0; i < 3; i++)
	{
		int j = (i + 1) % 3;
		if (this->lines_cross(p, outside_point, vertices[i], vertices[j]))
			crosses++;
	}

	return (crosses % 2 == 1);
}

/*
__host__ __device__ void swap(Triangle& a, Triangle& b)
{

	// Swap members
	Triangle tmp = Triangle();

	tmp.points = b.points;

	tmp.normal = b.normal;

	tmp.material = b.material;


	b.points = a.points;

	b.normal = a.normal;

	b.material = a.material;


	a.points = tmp.points;

	a.normal = tmp.normal;

	a.material = tmp.material;


	tmp.points = NULL;

	tmp.material = NULL;
}
*/

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
	const vec3 r_out = material->scatter(r_in, normal, rng);

	return Ray(-1, ixn_p, r_out);
}

__device__ vec3 Triangle::albedo(const vec3& p) const
{

	return vec3(1.f, 0.f, 1.f);
}

