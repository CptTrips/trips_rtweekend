#include "Mesh.cuh"


__host__ __device__ Mesh::Mesh(const Array<vec3>* const vertices, const Array<uint32_t>* const indices, const Material<CUDA_RNG>* const material) : vertices(vertices), indices(indices), material(material)
{

	assert(indices->size() % 3 == 0);

	// find bbox. parallelizable.

	vec3 min, max, cur;

	for (uint32_t i = 0; i < (vertices->size()); i++)
	{
		cur = (*vertices)[i];

		for (int j = 0; j < 3; j++)
		{
			min[j] = fminf(cur[j], min[j]);
			max[j] = fmaxf(cur[j], max[j]);
		}
	}

	construct_bbox(min, max);

	// Turn vertices and indices into triangles

	uint32_t size = indices->size() / 3;

	triangles = Array<TriangleView>(size);

	for (int i = 0; i < size; i++)
		triangles[i] = TriangleView(vertices, indices, i*3, material);
}

__host__ __device__ Mesh::~Mesh()
{
	if (vertices)
		delete vertices;

	if (indices)
		delete indices;

}

__device__ Intersection* Mesh::intersect(const Ray& r, float tmin, float tmax) const
{
	Intersection* ixn = NULL;

	if (intersect_bbox(r, tmin, tmax))
	{
		Intersection* temp_ixn;

		for (uint32_t i = 0; i < triangles.size(); i++)
		{
			
			temp_ixn = triangles[i].intersect(r, tmin, tmax);

			if (temp_ixn)
			{

				if (ixn)
					delete ixn;

				ixn = temp_ixn;

				tmax = ixn->t;

			}
		}
	}


	return ixn;
}

__device__ Ray Mesh::bounce(const vec3& r_in, const vec3& ixn_p, CUDA_RNG* rng) const
{
	printf("Error: Mesh::bounce!!");
	assert(false);
}

__device__ vec3 Mesh::albedo(const vec3& p) const
{
	printf("Error: Mesh::albedo!!");
	assert(false);

}

__device__ bool Mesh::intersect_bbox(const Ray& r, float tmin, float tmax) const
{

	Intersection* ixn;

	for (int i = 0; i < 12; i++)
	{

		ixn = bbox_triangles[i].intersect(r, tmin, tmax);

		if (ixn)
			delete ixn;
			return true;

	}

	return false;

}

__device__ void Mesh::construct_bbox(const vec3& min, const vec3& max)
{

	vec3 bbox_bounds[2] = { min, max };

	vec3 bbox_vertices[8];

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			for (int k = 0; k < 2; k++)
			{
				int corner_idx = k + 2 * j + 4 * i;

				bbox_vertices[corner_idx] = vec3(bbox_bounds[i].x(), bbox_bounds[j].y(), bbox_bounds[k].z());

			}
		}
	}

	vec3* bbox_tri_vertices[12] = 
	{
		new vec3[3]{ bbox_vertices[0], bbox_vertices[2], bbox_vertices[1] }
		,new vec3[3]{ bbox_vertices[2], bbox_vertices[3], bbox_vertices[1] }
		,new vec3[3]{ bbox_vertices[1], bbox_vertices[5], bbox_vertices[4] }
		,new vec3[3]{ bbox_vertices[0], bbox_vertices[1], bbox_vertices[4] }
		,new vec3[3]{ bbox_vertices[4], bbox_vertices[5], bbox_vertices[7] }
		,new vec3[3]{ bbox_vertices[4], bbox_vertices[7], bbox_vertices[6] }
		,new vec3[3]{ bbox_vertices[6], bbox_vertices[7], bbox_vertices[3] }
		,new vec3[3]{ bbox_vertices[3], bbox_vertices[2], bbox_vertices[6] }
		,new vec3[3]{ bbox_vertices[1], bbox_vertices[3], bbox_vertices[5] }
		,new vec3[3]{ bbox_vertices[1], bbox_vertices[7], bbox_vertices[5] }
		,new vec3[3]{ bbox_vertices[0], bbox_vertices[4], bbox_vertices[6] }
		,new vec3[3]{ bbox_vertices[0], bbox_vertices[6], bbox_vertices[2] }
	};

	bbox_triangles = Array<Triangle>(12);

	for (int i = 0; i < 12; i++)
	{
		bbox_triangles[i] = Triangle(bbox_tri_vertices[i], material);
		delete[] bbox_tri_vertices[i];
	}

}
