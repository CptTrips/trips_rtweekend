#include "GPURayTracer.cuh"
#define seed 1234

FrameBuffer* GPURayTracer::render(const int h, const int w, CUDAVisible** const scene, const int scene_size, const Camera& camera)
{
	gpu_scene = scene;

	this->scene_size = scene_size;

	ray_count = spp * h * w;

	// Allocate Frame Buffer
	h_fb = new FrameBuffer(h, w);

	// Send Camera
	cudaMalloc(&d_cam, sizeof(Camera));

	cudaMemcpy(d_cam, &camera, sizeof(Camera), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	
	// Generate rays on device
	generate_rays();

	cudaDeviceSynchronize();

	// Shade rays with kernel (one ray per kernel)
	// Inefficient!
	shade_rays();

	cudaDeviceSynchronize();

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


	// Free GPU memory
	// Scene pointers
	// Scene objects
	// Camera
	cudaFree(d_cam);
	cudaFree(ray_colours);

	return h_fb;
}

__global__ void colour_space(FrameBuffer* const fb)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < fb->h * fb->w)
	{
		int row = id / fb->w;
		int col = id % fb->w;

		vec3 colour = vec3(0.2f, float(row) / float(fb->w), float(col) / float(fb->h));

		fb->set_pixel(row, col, colour);
	}
}


void GPURayTracer::generate_rays()
{

	cudaMalloc(&rays, ray_count*sizeof(Ray));

	curandState *cr_state;

	cudaMallocManaged(&cr_state, sizeof(curandState));

	int threads = 512;

	int blocks = ray_count / threads + 1;


	std::cout << "generate_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_gen_rays<<<blocks, threads>>>(rays, ray_count, d_cam, h_fb, cr_state, spp);

	// Allocate ray colours
	cudaMalloc(&ray_colours, ray_count * sizeof(vec3));

}

__global__ void cuda_gen_rays(Ray* rays, const int ray_count, const Camera* const cam, const FrameBuffer* const fb, curandState* cr_state, const int spp)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < ray_count)
	{
		curand_init(seed, id, 0, cr_state);

		float dx = curand_uniform(cr_state);
		float dy = curand_uniform(cr_state);

		int row = id / spp / fb->w;

		float v = row / fb->h;

		int col = (id / spp) % fb->w;

		float u = col / fb->w;

		float x = (u - 0.5) * cam->vfov; // x goes from -vfov/2 to vfov/2
		float y = (v - 0.5) * cam->vfov * cam->aspect_ratio; // y goes from -w/(2h) to w/(2h)

		vec3 cam_space_ray_dir = vec3(x, y, cam->focus_distance);

		vec3 ray_dir = cam->orientation.T() * cam_space_ray_dir;

		rays[id] = Ray(cam->origin, ray_dir);
	}

}

void GPURayTracer::shade_rays()
{
	
	int threads = 512;

	int blocks = ray_count / threads + 1;

	std::cout << "shade_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_shade_ray << <blocks, threads>> > (rays, ray_colours, ray_count, gpu_scene, scene_size, max_bounce);
}

__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const int ray_count, CUDAVisible** const scene, const int scene_size, const int max_bounce)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < ray_count)
	{

		Ray ray = rays[id];

		ray_colours[id] = vec3(1.f, 1.f, 1.f);

		CUDA_RNG rng(seed, id);

		int bounce = 0;

		while (bounce < max_bounce)
		{
			Intersection* ixn_ptr = nearest_intersection(ray, scene, scene_size, 1e-12, FLT_MAX);
			
			if (ixn_ptr)
			{
				const CUDAVisible* const active_visible = ixn_ptr->visible;

				const vec3 ixn_p = ray.point_at(ixn_ptr->t);

				Ray scatter_ray = active_visible->bounce(ray.direction(), ixn_p, &rng);

				rays[id] = scatter_ray;

				ray_colours[id] *= active_visible->albedo(ixn_p);
			}

			else
			{
				ray_colours[id] *= draw_sky(ray);
				break;
			}

			bounce++;
		}

		if (bounce == max_bounce)
			ray_colours[id] = vec3(1.f, 0.f, 0.f);
	}
}


void GPURayTracer::render_rays()
{
	int threads = 512;
	int blocks = h_fb->h * h_fb->w / threads + 1;

	std::cout << "render_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_render_rays << <blocks, threads >> > (ray_colours, ray_count, h_fb, spp);
}

__global__ void cuda_render_rays(vec3* ray_colours, const int ray_count, FrameBuffer* fb, const int spp)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;

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

