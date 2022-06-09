#include "GPURayTracer.cuh"
#define my_cuda_seed 1234

FrameBuffer* GPURayTracer::render(const int h, const int w, CUDAVisible** const scene, const int scene_size, const Camera& camera)
{
	gpu_scene = scene;

	this->scene_size = scene_size;

	ray_count = spp * h * w;

	// Allocate Frame Buffer
	h_fb = new FrameBuffer(h, w);

	// Send Camera
	checkCudaErrors(cudaMalloc(&d_cam, sizeof(Camera)));

	checkCudaErrors(cudaMemcpy(d_cam, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();

	// Create rngs for each ray
	create_rngs();

	using milli = std::chrono::milliseconds;

	auto start = std::chrono::high_resolution_clock::now();

	// Generate rays on device
	generate_rays();

	// Shade rays with kernel (one ray per kernel)
	// Inefficient!
	shade_rays();

	checkCudaErrors(cudaFree(rays));

	cudaError_t cerr = cudaPeekAtLastError();
	if (cerr != cudaSuccess)
	{
		std::cerr << cerr << std::endl;
	}

	// Reduce colour for each pixel
	render_rays();

	/*
	int threads = 1024;
	int blocks = h * w / threads + 1;
	colour_space << <blocks, threads >> > (h_fb);
	*/

	cudaDeviceSynchronize();

	auto finish = std::chrono::high_resolution_clock::now();
	
	std::cout << "Frametime: "
		<< std::chrono::duration_cast<milli>(finish - start).count() << "ms"
		<< std::endl;

	// Free GPU memory
	// Scene pointers
	// Scene objects
	// Camera
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

	checkCudaErrors(cudaMalloc(&rngs, ray_count * sizeof(CUDA_RNG)));

	cudaDeviceSynchronize();

	uint32_t blocks = ray_count / threads + 1;

	cuda_create_rngs << <blocks, threads >> > (rngs, ray_count);

	cudaDeviceSynchronize();
}

__global__ void cuda_create_rngs(CUDA_RNG* const rngs, const uint32_t ray_count)
{

	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < ray_count)
		rngs[id] = CUDA_RNG(my_cuda_seed, id);


}

void GPURayTracer::generate_rays()
{
	// Allocate ray colours
	checkCudaErrors(cudaMalloc(&ray_colours, ray_count * sizeof(vec3)));

	// Allocate rays
	checkCudaErrors(cudaMalloc(&rays, ray_count*sizeof(Ray)));

	cudaDeviceSynchronize();

	uint32_t blocks = ray_count / threads + 1;

	std::cout << "generate_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_gen_rays<<<blocks, threads>>>(rays, ray_count, d_cam, h_fb, rngs, spp);

	cudaDeviceSynchronize();

}

__global__ void cuda_gen_rays(Ray* rays, const uint32_t ray_count, const Camera* const cam, const FrameBuffer* const fb, CUDA_RNG* const rngs, const int spp)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < ray_count)
	{
		CUDA_RNG rng = rngs[id];

		float dx = rng.sample();
		float dy = rng.sample();

		int row = id / spp / fb->w;

		float u = (float(row) + dy) / fb->h;

		int col = (id / spp) % fb->w;

		float v = (float(col) + dx) / fb->w;

		float x = (u - 0.5) * cam->vfov; // x goes from -vfov/2 to vfov/2
		float y = (v - 0.5) * cam->vfov * cam->aspect_ratio; // y goes from -w/(2h) to w/(2h)

		vec3 focus_offset = cam->aperture * (2 * vec3(rng.sample(), rng.sample(), 0) - vec3(1., 1., 0.));

		vec3 cam_space_ray_dir = vec3(x, y, cam->focus_distance) - focus_offset;

		vec3 ray_dir = cam->orientation.T() * cam_space_ray_dir;

		rays[id] = Ray(cam->origin + cam->orientation.T()*focus_offset, ray_dir);
	}

}

void GPURayTracer::shade_rays()
{
	
	uint32_t blocks = ray_count / threads + 1;

	std::cout << "shade_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_shade_ray << <blocks, threads>> > (rays, ray_colours, ray_count, gpu_scene, scene_size, max_bounce, rngs);

	cudaDeviceSynchronize();
}

__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const uint32_t ray_count, CUDAVisible** const scene, const int scene_size, const int max_bounce, CUDA_RNG* const rngs)
{

	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < ray_count)
	{

		Ray ray = rays[id];

		vec3 ray_colour = vec3(1.f, 1.f, 1.f);

		CUDA_RNG rng = rngs[id];

		int bounce = 0;

		Intersection* ixn_ptr;

		while (bounce < max_bounce)
		{
			ixn_ptr = nearest_intersection(ray, scene, scene_size, 1.e-12f, FLT_MAX);
			
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
			ray_colours[id] = vec3(0.f, 0.f, 0.f);
		else
			ray_colours[id] = ray_colour;
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


void GPURayTracer::render_rays()
{
	int blocks = h_fb->h * h_fb->w / threads + 1;

	std::cout << "render_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_render_rays << <blocks, threads >> > (ray_colours, ray_count, h_fb, spp);

	cudaDeviceSynchronize();
}

__global__ void cuda_render_rays(vec3* ray_colours, const uint32_t ray_count, FrameBuffer* fb, const int spp)
{

	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < fb->h * fb->w)
	{
		const int row = id / fb->w;
		const int col = id % fb->w;

		vec3 colour = vec3(0.f, 0.f, 0.f);

		for (int i = 0; i < spp; i++)
		{
			colour += ray_colours[i + col * spp + row * spp * fb->w];
		}

		colour /= spp;

		fb->set_pixel(row, col, colour);
	}
}


