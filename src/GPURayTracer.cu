#include "GPURayTracer.cuh"
#define my_cuda_seed 1234
#define DEBUG false

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

void GPURayTracer::terminateRays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices)
{

	uint32_t threads = max_threads;
	uint32_t blocks = p_activeRayIndices->size() / threads + 1;

	cuda_terminate_rays << <blocks, threads >> > (p_rayBuffer, p_activeRayIndices);

	checkCudaErrors(cudaDeviceSynchronize());
}

GPURayTracer::GPURayTracer() : ixnEngine()
{

	showDeviceProperties();

	increaseStackLimit();
}

FrameBuffer* GPURayTracer::render(const GPURenderProperties& render_properties, const Camera& camera)
{
	spp = render_properties.spp;

	ray_count = spp * render_properties.h * render_properties.w;

	max_bounce = render_properties.max_bounce;

	ixnEngine.minFreePath = render_properties.min_free_path;

	rays_per_batch = std::min(ray_count, (uint64_t)spp * (max_rays_per_batch / spp));

	// Allocate Frame Buffer
	h_fb = new FrameBuffer(render_properties.h, render_properties.w);

	// Send Camera
	checkCudaErrors(cudaMalloc(&d_cam, sizeof(Camera)));

	checkCudaErrors(cudaMemcpy(d_cam, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();

	// Allocate ray data (Ray, colour, rng)
	allocate_rays();

	UnifiedArray<vec3>* p_vertexBuffer = new UnifiedArray<vec3>(4);

	UnifiedArray<uint32_t>* p_indexBuffer = new UnifiedArray<uint32_t>(6);

	float floorSize = 1000.0f;

	(*p_vertexBuffer)[0] = vec3(-1.1, floorSize, floorSize);
	(*p_vertexBuffer)[1] = vec3(-1.1, floorSize, -floorSize);
	(*p_vertexBuffer)[2] = vec3(-1.1, -floorSize, floorSize);
	(*p_vertexBuffer)[3] = vec3(-1.1, -floorSize, -floorSize);

	(*p_indexBuffer)[0] = 1;
	(*p_indexBuffer)[1] = 0;
	(*p_indexBuffer)[2] = 3;
	(*p_indexBuffer)[3] = 0;
	(*p_indexBuffer)[4] = 2;
	(*p_indexBuffer)[5] = 3;

	UnifiedArray<Intersection>* p_triangleIntersectionBuffer = new UnifiedArray<Intersection>(p_rayBuffer->size());

	UnifiedArray<vec3>* p_triangleColourBuffer = new UnifiedArray<vec3>(p_indexBuffer->size());

	(*p_triangleColourBuffer)[0] = vec3(.8f, 0.8f, 0.6f);
	(*p_triangleColourBuffer)[1] = vec3(0.6f, 0.8f, 0.6f);

	UnifiedArray<CUDASphere>* p_sphereBuffer = new UnifiedArray<CUDASphere>(3);

	float bigRadius = 1000.f;

	(*p_sphereBuffer)[0] = CUDASphere{ vec3(0.0, 0., 1.2), 1.0, nullptr };
	(*p_sphereBuffer)[1] = CUDASphere{ vec3(0.0, 0., -1.2), 1.0, nullptr };
	(*p_sphereBuffer)[2] = CUDASphere{ vec3(0.0f, -bigRadius - 1.f, 0.f), bigRadius, nullptr };

	UnifiedArray<vec3>* p_sphereColourBuffer = new UnifiedArray<vec3>(p_sphereBuffer->size());

	(*p_sphereColourBuffer)[0] = vec3(0.4f, 0.8f, 1.f);
	(*p_sphereColourBuffer)[1] = vec3(0.8f, 0.4f, 1.f);
	(*p_sphereColourBuffer)[2] = vec3(0.8f, 0.8f, 1.f);

	UnifiedArray<Intersection>* p_sphereIntersectionBuffer = new UnifiedArray<Intersection>(p_rayBuffer->size());

	using milli = std::chrono::milliseconds;

	auto start = std::chrono::high_resolution_clock::now();

	for (uint64_t rayIDOffset = 0; rayIDOffset < ray_count; rayIDOffset += rays_per_batch)
	{

		std::cout << "Ray progress " << (float)rayIDOffset / (float)ray_count * 100 << "% " << rayIDOffset << " / " << ray_count << std::endl;

		generate_primary_rays(rayIDOffset);

		UnifiedArray<uint32_t>* p_activeRayIndices = resetActiveRays(p_rayBuffer->size());

		for (uint16_t bounce = 0; bounce < max_bounce; bounce++)
		{

			ixnEngine.run(p_rayBuffer, p_activeRayIndices, p_vertexBuffer, p_indexBuffer, p_sphereBuffer, p_triangleIntersectionBuffer, p_sphereIntersectionBuffer);

			colourRays(p_rayBuffer, p_activeRayIndices, p_triangleColourBuffer, p_sphereColourBuffer, p_triangleIntersectionBuffer, p_sphereIntersectionBuffer);

			p_activeRayIndices = gatherActiveRays(p_activeRayIndices, p_triangleIntersectionBuffer, p_sphereIntersectionBuffer);

			if (p_activeRayIndices->size() != 0)
				scatterRays(p_rayBuffer, p_activeRayIndices, p_vertexBuffer, p_indexBuffer, p_sphereBuffer, p_triangleIntersectionBuffer, p_sphereIntersectionBuffer);
		}

		terminateRays(p_rayBuffer, p_activeRayIndices);

		delete p_activeRayIndices;

		render_rays(rayIDOffset);

		std::cout << std::endl;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	auto finish = std::chrono::high_resolution_clock::now();
	
	std::cout << "Frametime: "
		<< std::chrono::duration_cast<milli>(finish - start).count() << "ms"
		<< std::endl;

	// Free GPU memory
	// Scene pointers
	// Scene objects
	// Camera
	delete p_rayBuffer;
	delete p_triangleIntersectionBuffer;
	delete p_sphereIntersectionBuffer;

	cudaFree(d_cam);
	cudaFree(rngs);

	return h_fb;
}

void GPURayTracer::allocate_rays()
{

	// Allocate rays
	p_rayBuffer = new UnifiedArray<Ray>(rays_per_batch);

	// RNG for each ray
	create_rngs();

	checkCudaErrors(cudaDeviceSynchronize());

}

void GPURayTracer::create_rngs()
{

	checkCudaErrors(cudaMalloc(&rngs, rays_per_batch * sizeof(CUDA_RNG)));

	cudaDeviceSynchronize();

	uint32_t threads = max_threads;
	uint32_t blocks = rays_per_batch / threads + 1;

	cuda_create_rngs << <blocks, threads >> > (rngs, rays_per_batch);

	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void cuda_create_rngs(CUDA_RNG* const rngs, const uint32_t rays_per_batch)
{

	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < rays_per_batch)
		rngs[id] = CUDA_RNG(my_cuda_seed, id);


}


void GPURayTracer::generate_primary_rays(const uint64_t ray_offset_index)
{

	uint32_t threads = max_threads;
	uint32_t blocks = rays_per_batch / threads + 1;

	std::cout << "generate_primary_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_gen_rays<<<blocks, threads>>>(&(*p_rayBuffer)[0], ray_count, rays_per_batch, ray_offset_index, d_cam, h_fb, rngs, spp);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaGetLastError());

}

void GPURayTracer::showDeviceProperties()
{

	cout << "Device Properties" << endl;

	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0); // assume one CUDA device

	cout << "Max Grid Size: " << prop.maxGridSize[0] << "x " << prop.maxGridSize[1] << "y " << prop.maxGridSize[2] << "z " << endl;
	cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << endl;
	cout << "Shared Mem Per Block: " << prop.sharedMemPerBlock << endl;

	cout << endl;
}

void GPURayTracer::increaseStackLimit()
{

	size_t stack_limit;

	checkCudaErrors(cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize));

	std::cout << "Default stack limit: " << stack_limit << std::endl;

	checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 8*stack_limit));

	checkCudaErrors(cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize));

	std::cout << "New stack limit: " << stack_limit << std::endl;

	cout << endl;
}

__global__ void cuda_gen_rays(Ray* rays, const uint64_t ray_count, const uint64_t rays_per_batch, const uint64_t ray_offset_index, const Camera* const cam, const FrameBuffer* const fb, CUDA_RNG* const rngs, const int spp)
{
	uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	uint64_t ray_id = thread_id + ray_offset_index;

	if ((ray_id < ray_count) && (thread_id < rays_per_batch))
	{
		CUDA_RNG rng = rngs[thread_id];

		float dx = rng.sample();
		float dy = rng.sample();

		int row = ray_id / (uint64_t)(spp * fb->w);

		float u = (float(row) + dy) / fb->h;

		int col = (ray_id / (uint64_t)spp) % fb->w;

		float v = (float(col) + dx) / fb->w;

		float x = (u - 0.5) * cam->vfov; // x goes from -vfov/2 to vfov/2
		float y = (v - 0.5) * cam->vfov * cam->aspect_ratio; // y goes from -w/(2h) to w/(2h)

		vec3 focus_offset = cam->aperture * (2 * vec3(rng.sample(), rng.sample(), 0) - vec3(1., 1., 0.));

		vec3 cam_space_ray_dir = vec3(x, y, cam->focus_distance) - focus_offset;

		vec3 ray_dir = cam->orientation.T() * cam_space_ray_dir;

		rays[thread_id] = Ray(ray_id, cam->origin + cam->orientation.T()*focus_offset, ray_dir);
	}

}

void GPURayTracer::colourRays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_triangleColurBuffer, UnifiedArray<vec3>* p_sphereColourBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer)
{


	uint32_t threads = max_threads;
	uint32_t blocks = rays_per_batch / threads + 1;

	cuda_colour_rays << <blocks, threads >> > (p_rayBuffer, p_activeRayIndices, p_triangleColurBuffer, p_sphereColourBuffer, p_triangleIntersectionBuffer, p_sphereIntersectionBuffer);

	checkCudaErrors(cudaDeviceSynchronize());
}

void GPURayTracer::render_rays(const uint64_t ray_offset_index)
{

	uint32_t threads = max_threads;

	int pixel_start_idx = ray_offset_index / (uint64_t)spp;

	int pixel_end_idx = pixel_start_idx + rays_per_batch / spp; // not including this index

	pixel_end_idx = std::min(pixel_end_idx, h_fb->h * h_fb->w);

	int pixel_batch_size = pixel_end_idx - pixel_start_idx;

	int blocks = pixel_batch_size / threads + 1;

	std::cout << "render_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_render_rays << <blocks, threads >> > (pixel_start_idx, pixel_end_idx, p_rayBuffer, h_fb, spp);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaGetLastError());
}

UnifiedArray<uint32_t>* GPURayTracer::resetActiveRays(const uint32_t& bufferSize)
{

	UnifiedArray<uint32_t>* p_activeRayIndices = new UnifiedArray<uint32_t>(bufferSize);

	uint32_t threads = max_threads;
	uint32_t blocks = rays_per_batch / threads + 1;

	cuda_reset_active_rays << <blocks, threads >> > (p_activeRayIndices);

	checkCudaErrors(cudaDeviceSynchronize());

	return p_activeRayIndices;
}

UnifiedArray<uint32_t>* GPURayTracer::gatherActiveRays(UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer)
{

	uint32_t length = p_activeRayIndices->size();

	// Create 0, 1 mask for intersections
	UnifiedArray<uint8_t>* p_mask = new UnifiedArray<uint8_t>(length);

	// Create scan of mask
	UnifiedArray<uint32_t>* p_scan = new UnifiedArray<uint32_t>(length);

	uint32_t activeRayCount = 0;

	for (uint32_t i = 0; i < length; i++)
	{

		uint8_t isActive = (isinf((*p_sphereIntersectionBuffer)[i].t) && isinf((*p_triangleIntersectionBuffer)[i].t)) ? 0 : 1;

		(*p_scan)[i] = activeRayCount;

		activeRayCount += isActive;

		(*p_mask)[i] = isActive;
	}

	// Create new active ray index array
	UnifiedArray<uint32_t>* p_newActiveRayIndices = new UnifiedArray<uint32_t>(activeRayCount);

	for (uint32_t i = 0; i < length; i++)
	{

		if ((*p_mask)[i] == 1)
			(*p_newActiveRayIndices)[(*p_scan)[i]] = (*p_activeRayIndices)[i];
	}

	delete p_mask;
	delete p_scan;
	delete p_activeRayIndices;
	
	return p_newActiveRayIndices;
}

void GPURayTracer::scatterRays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_vertexBuffer, UnifiedArray<uint32_t>* p_indexBuffer, UnifiedArray<CUDASphere>* p_sphereBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer)
{

	
	uint32_t threads = max_threads;
	uint32_t blocks = p_activeRayIndices->size() / threads + 1;

	cuda_scatter_rays << <blocks, threads >> > (
		p_rayBuffer,
		p_activeRayIndices,
		p_vertexBuffer,
		p_indexBuffer,
		p_sphereBuffer,
		p_triangleIntersectionBuffer,
		p_sphereIntersectionBuffer,
		rngs
	);

	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void cuda_render_rays(const int pixel_start_idx, const int pixel_end_idx, UnifiedArray<Ray>* p_rayBuffer, FrameBuffer* fb, const int spp)
{

	const uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

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

	if (THREAD_ID > p_activeRayIndices->size())
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

void GPURayTracer::shade_rays(const uint64_t ray_offset_index)
{
	
	uint32_t threads = 512;

	uint32_t blocks = rays_per_batch / threads + 1;

	std::cout << "shade_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	size_t stack_size;

	checkCudaErrors(cudaThreadGetLimit(&stack_size, cudaLimitStackSize));

	cuda_shade_ray << <blocks, threads >> > (rays, ray_colours, ray_count, rays_per_batch, ray_offset_index, visibles, max_bounce, min_free_path, rngs);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaGetLastError());
}

__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const uint64_t ray_count, const uint64_t rays_per_batch, const uint64_t ray_offset_index, const UnifiedArray<CUDAVisible*>* const visibles, const int max_bounce, const float min_free_path, CUDA_RNG* const rngs)
{

	uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	uint64_t ray_id = thread_id + ray_offset_index;

	if ((ray_id < ray_count) && (thread_id < rays_per_batch))
	{

		Ray ray = rays[thread_id];

		vec3 ray_colour = vec3(1.f, 1.f, 1.f);

		CUDA_RNG rng = rngs[thread_id];

		int bounce = 0;

		Intersection* ixn_ptr;

		while (bounce < max_bounce)
		{

			ixn_ptr = nearest_intersection(ray, visibles, min_free_path, FLT_MAX);

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


		if (bounce == max_bounce)
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
