#include "GPURayTracer.cuh"
#define my_cuda_seed 1234
#define DEBUG false

FrameBuffer* GPURayTracer::render(const GPURenderProperties& render_properties, CUDAVisible** const scene, const int scene_size, const Camera& camera)
{
	gpu_scene = scene;

	this->scene_size = scene_size;

	spp = render_properties.spp;

	ray_count = spp * render_properties.h * render_properties.w;

	max_bounce = render_properties.max_bounce;

	rays_per_batch = std::min(ray_count, (uint64_t)spp * (max_rays_per_batch / spp));

	// Allocate Frame Buffer
	h_fb = new FrameBuffer(render_properties.h, render_properties.w);

	// Send Camera
	checkCudaErrors(cudaMalloc(&d_cam, sizeof(Camera)));

	checkCudaErrors(cudaMemcpy(d_cam, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();

	// Allocate ray data (Ray, colour, rng)
	allocate_rays();

	using milli = std::chrono::milliseconds;

	auto start = std::chrono::high_resolution_clock::now();

	for (uint64_t i = 0; i < ray_count; i += rays_per_batch)
	{

		std::cout << "Ray progress " << (float)i / (float)ray_count*100 << "% " << i << " / " << ray_count << std::endl;

		// Generate rays on device
		generate_rays(i);

		// Shade rays with kernel (one ray per kernel)
		// Inefficient!
		shade_rays(i);

		// Reduce colour for each pixel
		render_rays(i);

		std::cout << std::endl;
	}

	/*
	int threads = 1024;
	int blocks = h * w / threads + 1;
	colour_space << <blocks, threads >> > (h_fb);
	*/

	checkCudaErrors(cudaDeviceSynchronize());

	auto finish = std::chrono::high_resolution_clock::now();
	
	std::cout << "Frametime: "
		<< std::chrono::duration_cast<milli>(finish - start).count() << "ms"
		<< std::endl;

	// Free GPU memory
	// Scene pointers
	// Scene objects
	// Camera
	checkCudaErrors(cudaFree(rays));
	cudaFree(d_cam);
	cudaFree(ray_colours);
	cudaFree(rngs);

	return h_fb;
}

__global__ void colour_space(FrameBuffer* const fb)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < fb->h * fb->w)
	{
		int row = id / fb->w;
		int col = id % fb->w;

		vec3 colour = vec3(0.2f, float(row) / float(fb->w), float(col) / float(fb->h));

		fb->set_pixel(row, col, colour);
	}
}

void GPURayTracer::create_rngs()
{

	checkCudaErrors(cudaMalloc(&rngs, rays_per_batch * sizeof(CUDA_RNG)));

	cudaDeviceSynchronize();

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

void GPURayTracer::allocate_rays()
{
	// Allocate ray colours
	checkCudaErrors(cudaMalloc(&ray_colours, rays_per_batch * sizeof(vec3)));

	// Allocate rays
	checkCudaErrors(cudaMalloc(&rays, rays_per_batch * sizeof(Ray)));

	// RNG for each ray
	create_rngs();

	checkCudaErrors(cudaDeviceSynchronize());

}

void GPURayTracer::generate_rays(const uint64_t ray_offset_index)
{

	uint32_t blocks = rays_per_batch / threads + 1;

	std::cout << "generate_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_gen_rays<<<blocks, threads>>>(rays, ray_count, rays_per_batch, ray_offset_index, d_cam, h_fb, rngs, spp);

	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());

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

		rays[thread_id] = Ray(cam->origin + cam->orientation.T()*focus_offset, ray_dir);
	}

}

void GPURayTracer::shade_rays(const uint64_t ray_offset_index)
{
	
	uint32_t blocks = rays_per_batch / threads + 1;

	std::cout << "shade_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_shade_ray << <blocks, threads>> > (rays, ray_colours, ray_count, rays_per_batch, ray_offset_index, gpu_scene, scene_size, max_bounce, rngs);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaGetLastError());
	//cudaDeviceSynchronize();
}

__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const uint64_t ray_count, const uint64_t rays_per_batch, const uint64_t ray_offset_index, CUDAVisible** const scene, const int scene_size, const int max_bounce, CUDA_RNG* const rngs)
{

	uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	uint64_t ray_id = thread_id + ray_offset_index;

	if ((ray_id < ray_count) && (thread_id < rays_per_batch))
	{

		if (DEBUG) printf("Ray id %u\n", ray_id);

		Ray ray = rays[thread_id];

		vec3 ray_colour = vec3(1.f, 1.f, 1.f);

		CUDA_RNG rng = rngs[thread_id];

		int bounce = 0;

		Intersection* ixn_ptr;

		while (bounce < max_bounce)
		{
			ixn_ptr = nearest_intersection(ray, scene, scene_size, 1.e-12f, FLT_MAX);

			if (DEBUG) printf("%u: bounce %i intersections computed\n", ray_id, bounce);
			
			if (ixn_ptr)
			{
				const CUDAVisible* const active_visible = ixn_ptr->visible;

				if (DEBUG) printf("%u: intersection found\n", ray_id);

				const vec3 ixn_p = ray.point_at(ixn_ptr->t);
				if (DEBUG) printf("%u: ixn pt %4.2f %4.2f %4.2f\n", ray_id, ixn_p.x(), ixn_p.y(), ixn_p.z());

				ray = active_visible->bounce(ray.direction(), ixn_p, &rng);
				if (DEBUG) printf("%u: scatter dir %4.2f %4.2f %4.2f\n", ray_id, ray.d.x(), ray.d.y(), ray.d.z());

				ray_colour *= active_visible->albedo(ixn_p);
				if (DEBUG) printf("%u: albedo %4.2f %4.2f %4.2f\n", ray_id, ray_colour.x(), ray_colour.y(), ray_colour.z());

				delete ixn_ptr;

				if (DEBUG) printf("%u: intersection consumed\n", ray_id);
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

__device__ Intersection* nearest_intersection(const Ray& ray, CUDAVisible** const scene, const int scene_size, const float tmin, const float tmax)
{
	Intersection* temp_ixn;

	Intersection* ixn = NULL;

	float current_closest = tmax;

	for (int i = 0; i < scene_size; i++)
	{
		
		temp_ixn = scene[i]->intersect(ray, tmin, current_closest);

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


__device__ vec3 draw_sky(const Ray& ray)
{

  vec3 unit_dir = normalise(ray.direction());
  float t = 0.5f*(unit_dir.y() + 1.0f);
  return (1.0 - t)*vec3(1.f, 1.f, 1.f) + t*vec3(0.5f, 0.7f, 1.0f);

}


void GPURayTracer::render_rays(const uint64_t ray_offset_index)
{

	int pixel_start_idx = ray_offset_index / (uint64_t)spp;

	int pixel_end_idx = pixel_start_idx + rays_per_batch / spp; // not including this index

	pixel_end_idx = std::min(pixel_end_idx, h_fb->h * h_fb->w);

	int pixel_batch_size = pixel_end_idx - pixel_start_idx;

	int blocks = pixel_batch_size / threads + 1;

	std::cout << "render_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_render_rays << <blocks, threads >> > (pixel_start_idx, pixel_end_idx, ray_colours, h_fb, spp);

	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());
	//cudaDeviceSynchronize();
}

__global__ void cuda_render_rays(const int pixel_start_idx, const int pixel_end_idx, vec3* ray_colours, FrameBuffer* fb, const int spp)
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
			colour += ray_colours[thread_id * spp + i];
		}

		colour /= spp;

		fb->set_pixel(row, col, colour);
	}
}


