#include "GPURayTracer.cuh"
#define seed 1234


FrameBuffer* GPURayTracer::render(const int h, const int w, const std::vector<std::unique_ptr<Visible>>& scene, const Camera& camera)
{
	ray_count = spp * h * w;

	// Send scene to device
	send_scene(scene);

	// Allocate Frame Buffer
	alloc_framebuffer(h, w);

	// Send Camera
	cudaMalloc(&d_cam, sizeof(Camera));

	cudaMemcpy(d_cam, &camera, sizeof(Camera), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	
	// Generate rays on device
	// (Rays know which pixel they originated from)
	generate_rays();

	cudaDeviceSynchronize();

	// Shade rays with kernel (one ray per kernel)
	// Inefficient!
	shade_rays();

	cudaDeviceSynchronize();

	cudaFree(rays);

	// Reduce colour for each pixel
	render_rays();

	cudaDeviceSynchronize();

	// Send framebuffer back to GPU
	cudaMemcpy(h_fb->buffer, d_fb->buffer, 3 * h_fb->h * h_fb->w * sizeof(char), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	// Free GPU memory
	// Scene pointers
	// Scene objects
	// Camera
	cudaFree(d_cam);
	cudaFree(d_fb->buffer);
	cudaFree(d_fb);
	cudaFree(gpu_scene);
	cudaFree(ray_colours);

	return h_fb;
}

void GPURayTracer::send_scene(const std::vector<std::unique_ptr<Visible>>& scene)
{

	scene_size = scene.size();
	// Allocate space for gpu_scene (pointers to visible)
	cudaMallocManaged(&gpu_scene, scene_size * sizeof(CUDAVisible*));

	// Iterate the scene object
	CUDAVisible** v_ptr = gpu_scene;
	for (const auto& v : scene)
	{
		// Copy the visible to GPU

		*v_ptr = v->to_device();

		v_ptr++;
	}

}


void GPURayTracer::alloc_framebuffer(const int h, const int w)
{

	// Allocate host memory
	h_fb = new FrameBuffer(h, w);

	// Allocate device memory
	cudaMallocManaged(&d_fb, sizeof(FrameBuffer));

	// Copy host fb to device
	cudaMemcpy(d_fb, h_fb, sizeof(FrameBuffer), cudaMemcpyHostToDevice);

	// Allocate device buffer memory
	char* d_fb_buffer;
	cudaMallocManaged(&d_fb_buffer, 3 * h * w * sizeof(char));

	// Point device fb to buffer address
	d_fb->buffer = d_fb_buffer;
	//checkCudaErrors(cudaMemcpy(&(d_fb->buffer), &d_fb_buffer, sizeof(char*), cudaMemcpyDeviceToDevice));
}


void GPURayTracer::generate_rays()
{

	cudaMalloc(&rays, ray_count*sizeof(Ray));

	curandState *cr_state;

	cudaMallocManaged(&cr_state, sizeof(curandState));

	cuda_gen_rays<<<ray_count / 2048 + 1, 2048>>>(rays, ray_count, d_cam, d_fb, cr_state, spp);

	cudaMalloc(&ray_colours, ray_count * sizeof(vec3));

}

void GPURayTracer::shade_rays()
{
	cuda_shade_ray << <ray_count / 2048 + 1, 2048 >> > (rays, ray_colours, ray_count, gpu_scene, scene_size, max_bounce);
}


void GPURayTracer::render_rays()
{
	cuda_render_rays << <h_fb->h * h_fb->w / 2048, 2048 >> > (ray_colours, ray_count, d_fb, spp);
}

__global__ void cuda_render_rays(vec3* ray_colours, const int ray_count, FrameBuffer* fb, const int spp)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < fb->h * fb->w)
	{
		const int row = id / fb->h;
		const int col = id % fb->w;

		vec3 colour = vec3(0.f, 0.f, 1.f);

		for (int i = 0; i < spp; i++)
		{
			colour += ray_colours[i + col * spp + row * spp * fb->w];
		}

		fb->set_pixel(row, col, colour);
	}
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

		int col = (id / spp) % fb->h;

		float u = col / fb->w;

		float x = (u - 0.5) * cam->vfov; // x goes from -vfov/2 to vfov/2
		float y = (v - 0.5) * cam->vfov * cam->aspect_ratio; // y goes from -w/(2h) to w/(2h)

		vec3 cam_space_ray_dir = vec3(x, y, cam->focus_distance);

		vec3 ray_dir = cam->orientation.T() * cam_space_ray_dir;

		rays[id] = Ray(cam->origin, ray_dir);
	}

}

__device__ Intersection* nearest_intersection(const Ray& ray, CUDAVisible** scene, const int scene_size, const float tmin, const float tmax)
{
	Intersection* temp_ixn;

	Intersection* ixn = NULL;

	float current_closest = tmax;

	for (int i = 0; i < scene_size; i++)
	{
		const CUDAVisible* v = scene[i];
		
		temp_ixn = v->intersect(ray, tmin, current_closest);

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


__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const int ray_count, CUDAVisible** scene, const int scene_size, const int max_bounce)
{

	int id = threadIdx.x + blockIdx.x + blockDim.x;

	if (id < ray_count)
	{

		Ray ray = rays[id];

		ray_colours[id] = vec3(0.f, 1.f, 0.f);

		int bounce = 0;

		while (bounce < max_bounce)
		{
			Intersection* ixn_ptr = nearest_intersection(ray, scene, scene_size, 1e-12, FLT_MAX);
			
			if (ixn_ptr)
			{
				const Material* const active_material = ixn_ptr->material;

				Ray scatter_ray;

				active_material->bounce(ray, *ixn_ptr, scatter_ray);

				rays[id] = scatter_ray;

				ray_colours[id] *= active_material->albedo;
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