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
	printf("Index array pointer: %p\n", this->index_array);
	printf("First triangle indices %d %d %d\n", (*this->index_array)[0], (*this->index_array)[1], (*this->index_array)[2]);
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

	if (ixn)
	{

		float t = ixn->t;

		delete ixn;

		ixn = new Intersection(t, this);

	}

	return ixn;

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

__device__ void TriangleView::print()
{
	printf("Index offset: %d", index_array_offset);
	printf("Index array size: %d", index_array->size());
}


__device__ Triangle TriangleView::construct_triangle() const
{
	/*
	const uint32_t indices[] = { (*index_array)[index_array_offset], (*index_array)[index_array_offset + 1], (*index_array)[index_array_offset + 2] };

	if (indices[2] > vertex_array->size())
	{
		printf("Index array pointer: %p\n", this->index_array);
		printf("First triangle indices %d %d %d\n", (*this->index_array)[0], (*this->index_array)[1], (*this->index_array)[2]);
		printf("Index array offset: %d\nIndex array size: %d\n", index_array_offset, index_array->size());
		printf("Vertex array %p\nVertex array index %d\n", vertex_array, indices[2]);
	}
	const vec3 points[] = { (*vertex_array)[indices[0]], (*vertex_array)[indices[1]], (*vertex_array)[indices[2]] };
	*/

	//printf("Offset %d / Size %d\n", index_array_offset, index_array->size());

	unsigned int index_0 = (*index_array)[index_array_offset];
	unsigned int index_1 = (*index_array)[index_array_offset + 1];
	unsigned int index_2 = (*index_array)[index_array_offset + 2];

	//printf("(%d, %d, %d) / %d\n", index_0, index_1, index_2, vertex_array->size());

	const vec3 points[3] = { (*vertex_array)[index_0], (*vertex_array)[index_1], (*vertex_array)[index_2] };

	Triangle t(points, material);

	return t;
}
