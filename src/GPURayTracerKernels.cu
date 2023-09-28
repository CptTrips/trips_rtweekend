#include "GPURayTracerKernels.cuh"

#define THREAD_ID threadIdx.x + blockIdx.x * blockDim.x
#define my_cuda_seed 1234

__global__ void cuda_scatter_rays(
	UnifiedArray<Ray>* p_rayBuffer,
	UnifiedArray<uint32_t>* p_activeRayIndices,
	UnifiedArray<vec3>* p_vertexBuffer,
	UnifiedArray<uint32_t>* p_indexBuffer,
	UnifiedArray<CUDASphere>* p_sphereBuffer,
	UnifiedArray<Intersection>* p_triangleIntersectionBuffer,
	UnifiedArray<Intersection>* p_sphereIntersectionBuffer,
	//UnifiedArray<Material<CUDA_RNG>>* p_materialBuffer,
	CUDA_RNG* const rngs
)
{

	if (THREAD_ID >= p_activeRayIndices->size())
		return;

	uint32_t rayID = (*p_activeRayIndices)[THREAD_ID];

	if ((*p_triangleIntersectionBuffer)[rayID].id == -1 && (*p_sphereIntersectionBuffer)[rayID].id == -1)
		return;

	Ray* p_ray = &(*p_rayBuffer)[rayID];

	CUDA_RNG& rng = rngs[rayID];

	Intersection ixn = ((*p_triangleIntersectionBuffer)[rayID].t < (*p_sphereIntersectionBuffer)[rayID].t) ? (*p_triangleIntersectionBuffer)[rayID] : (*p_sphereIntersectionBuffer)[rayID];

	if (ixn.normal.length() == 0.f)
		printf("Bad ixn normal %d\n", rayID);

	Material<CUDA_RNG> material(1.0f, 0.0f, 0.f, 0.0f, 1.f);

	p_ray->o = p_ray->point_at(ixn.t);
	p_ray->d = material.scatter(p_ray->d, ixn.normal, &rngs[rayID]);
}

__global__ void cuda_terminate_rays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activateRayIndices)
{

	if (THREAD_ID >= p_activateRayIndices->size())
		return;

	uint32_t rayID = (*p_activateRayIndices)[THREAD_ID];

	(*p_rayBuffer)[rayID].colour = vec3(0.f, 0.f, 0.f);
}

__global__ void cuda_is_active(UnifiedArray<uint32_t>* p_mask, UnifiedArray<Intersection>* p_triIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer)
{

	if (THREAD_ID >= p_triIntersectionBuffer->size())
		return;

	(*p_mask)[THREAD_ID] = (isinf((*p_sphereIntersectionBuffer)[THREAD_ID].t) && isinf((*p_triIntersectionBuffer)[THREAD_ID].t)) ? 0 : 1;
}

__global__ void cuda_create_rngs(UnifiedArray<CUDA_RNG>* const rngs)
{

	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < rngs->size())
		(*rngs)[id] = CUDA_RNG(my_cuda_seed, id);
}


__device__ ImagePoint subPixel(const uint64_t rayID, const FrameBuffer* const fb, const uint64_t spp, CUDA_RNG& rng)
{

	float dx = rng.sample();
	float dy = rng.sample();

	int row = rayID / (uint64_t)(spp * fb->w);

	float u = (float(row) + dy) / fb->h;

	int col = (rayID / (uint64_t)spp) % fb->w;

	float v = (float(col) + dx) / fb->w;

	return { u, v };
}

__global__ void cuda_gen_rays(Ray* rays, const uint64_t rayCount, const uint64_t raysPerBatch, const uint64_t ray_offset_index, const Camera* const cam, const FrameBuffer* const fb, CUDA_RNG* const rngs, const uint64_t spp)
{
	uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	uint64_t rayID = thread_id + ray_offset_index;

	if ((rayID < rayCount) && (thread_id < raysPerBatch))
	{
		CUDA_RNG rng = rngs[thread_id];

		ImagePoint p { subPixel(rayID, fb, spp, rng) };

		rays[thread_id] = cam->castRay(p, rng);
	}

}

__global__ void cuda_render_rays(const int pixel_start_idx, const int pixel_end_idx, UnifiedArray<Ray>* p_rayBuffer, FrameBuffer* fb, const uint64_t spp)
{

	const uint64_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	const uint32_t pixel_id = pixel_start_idx + thread_id;

	// Could dispatch a reduce kernel here
	if (pixel_id < pixel_end_idx)
	{
		const uint32_t row = pixel_id / fb->w;
		const uint32_t col = pixel_id % fb->w;

		vec3 colour = vec3(0.f, 0.f, 0.f);

		for (uint64_t i = 0; i < spp; i++)
		{
			colour += (*p_rayBuffer)[thread_id * spp + i].colour;
			//colour += 0.5 * (vec3(1.f, 1.f, 1.f) + (*p_rayBuffer)[thread_id * spp + i].d);
		}

		colour /= spp;

		// Gamma correction
		colour = gamma_correction(colour);

		fb->set_pixel(row, col, colour);
	}
}

__device__ vec3 gamma_correction(const vec3& col_in)
{

	return vec3(sqrtf(col_in.r()), sqrtf(col_in.g()), sqrtf(col_in.b()));
}

__host__ __device__ vec3 draw_sky(const Ray& ray)
{

  vec3 unit_dir = normalise(ray.direction());
  float t = 0.5f*(unit_dir.y() + 1.0f);
  return (1.0 - t)*vec3(1.f, 1.f, 1.0f) + t*vec3(0.5f, 0.7f, 1.0f);

}

__global__ void cuda_reset_active_rays(UnifiedArray<uint32_t>* p_activeRayIndices)
{

	if (THREAD_ID < p_activeRayIndices->size())
		(*p_activeRayIndices)[THREAD_ID] = THREAD_ID;
}

__global__ void cuda_colour_rays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_triangleColourBuffer, UnifiedArray<vec3>* p_sphereColourBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer)
{

	if (THREAD_ID >= p_activeRayIndices->size())
		return;

	uint32_t rayID = (*p_activeRayIndices)[THREAD_ID];

	Ray * p_ray = &((*p_rayBuffer)[rayID]);

	Intersection triangleIxn = (*p_triangleIntersectionBuffer)[rayID];

	Intersection sphereIxn = (*p_sphereIntersectionBuffer)[rayID];

	if (triangleIxn.t < sphereIxn.t)
		p_ray->colour *= (*p_triangleColourBuffer)[triangleIxn.id];
	else if (sphereIxn.t < triangleIxn.t)
		p_ray->colour *= (*p_sphereColourBuffer)[sphereIxn.id];
	else
		p_ray->colour *= draw_sky(*p_ray);
}

/*
__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const uint64_t rayCount, const uint64_t raysPerBatch, const uint64_t ray_offset_index, const UnifiedArray<CUDAVisible*>* const visibles, const int maxBounce, const float minFreePath, CUDA_RNG* const rngs)
{

	uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	uint64_t ray_id = thread_id + ray_offset_index;

	if ((ray_id < rayCount) && (thread_id < raysPerBatch))
	{

		Ray ray = rays[thread_id];

		vec3 ray_colour = vec3(1.f, 1.f, 1.f);

		CUDA_RNG rng = rngs[thread_id];

		int bounce = 0;

		Intersection* ixn_ptr;

		while (bounce < maxBounce)
		{

			ixn_ptr = nearest_intersection(ray, visibles, minFreePath, FLT_MAX);

			if (ixn_ptr)
			{
				const CUDAVisible* const active_visible = ixn_ptr->visible;

				const vec3 ixn_p = ray.point_at(ixn_ptr->t);

				ray = active_visible->bounce(ray.direction(), ixn_p, &rng);

				ray_colour *= active_visible->albedo(ixn_p);

				delete ixn_ptr;
			}

			else
			{
				ray_colour *= draw_sky(ray);
				break;
			}

			bounce++;
		}


		if (bounce == maxBounce)
			ray_colours[thread_id] = vec3(0.f, 0.f, 0.f);
		else
			ray_colours[thread_id] = ray_colour;

		if (DEBUG) printf("%u: ray colour assigned %4.2f %4.2f %4.2f\n", ray_id, ray_colours[thread_id].r(), ray_colours[thread_id].g(), ray_colours[thread_id].b());
	}
}

__device__ Intersection* nearest_intersection(const Ray& ray, const UnifiedArray<CUDAVisible*>* const visibles, const float tmin, const float tmax)
{
	Intersection* temp_ixn;

	Intersection* ixn = NULL;

	float current_closest = tmax;

	for (int i = 0; i < visibles->size(); i++)
	{
		
		temp_ixn = (*visibles)[i]->intersect(ray, tmin, current_closest);

		if (temp_ixn)
		{
			current_closest = temp_ixn->t;

			if (ixn)
				delete ixn;

			ixn = temp_ixn;
		}
	}

	return ixn;
}

*/