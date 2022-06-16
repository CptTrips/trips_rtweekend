#include "TriangleView.cuh"

TriangleView::TriangleView()
{
	vertex_array = NULL;
	material = NULL;
	index_array = NULL;
	index_array_offset = 0;
}

TriangleView::TriangleView(
	const Array<vec3>* const vertex_array
	,const Array<uint32_t>* const index_array
	,const uint32_t& index_0
	,const Material<CUDA_RNG>* const material
) : vertex_array(vertex_array), index_array(index_array), index_array_offset(index_0), material(material) 
{
}

__host__ __device__ TriangleView::TriangleView(const TriangleView& tv)
	: vertex_array(tv.vertex_array), index_array(index_array), index_array_offset(tv.index_array_offset), material(tv.material) 
{
}

__host__ __device__ TriangleView& TriangleView::operator=(const TriangleView& tv)
{

	if (this == &tv)
		return *this;

	vertex_array = tv.vertex_array;

	index_array = tv.index_array;

	index_array_offset = tv.index_array_offset;

	material = tv.material;

	return *this;

}

__host__ __device__ TriangleView::~TriangleView()
{

}

__device__ Intersection* TriangleView::intersect(const Ray& r, float tmin, float tmax) const
{

	Triangle triangle = construct_triangle();

	Intersection* ixn = triangle.intersect(r, tmin, tmax);

	Intersection* new_ixn = NULL;

	if (ixn)
	{
		new_ixn = new Intersection(ixn->t, this);

		delete ixn;

	}

	return new_ixn;

}

__device__ Ray TriangleView::bounce(const vec3& r_in, const vec3& ixn_p, CUDA_RNG* rng) const
{
	Triangle triangle = construct_triangle();

	return triangle.bounce(r_in, ixn_p, rng);
}

__device__ vec3 TriangleView::albedo(const vec3& p) const
{
	return material->albedo;
}


__device__ Triangle TriangleView::construct_triangle() const
{
	const uint32_t indices[3] = { (*index_array)[index_array_offset], (*index_array)[index_array_offset + 1], (*index_array)[index_array_offset + 2] };

	const vec3 points[3] = { (*vertex_array)[indices[0]], (*vertex_array)[indices[1]], (*vertex_array)[indices[2]] };

	return Triangle(points, material);
}
