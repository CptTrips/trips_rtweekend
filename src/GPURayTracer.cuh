#pragma once
#include "device_launch_parameters.h"
#include "visibles/sphere.cuh"
#include "FrameBuffer.cuh"
#include <vector>
#include "Camera.h"
#include "curand_kernel.h"
#include "ray.cuh"
#include "Error.cuh"
#include "rand.h"
#include <chrono>


class GPURayTracer
{

	const int threads = 512;

	// Maximum number of rays to dispatch at once (GPU memory limited)
	const uint32_t max_ray_batch = 22 * 1e6;

	const int spp;

	const int max_bounce;

	uint64_t ray_count;

	int scene_size;

	CUDAVisible** gpu_scene;

	FrameBuffer* h_fb;

	Camera* d_cam;

	Ray* rays;

	vec3* ray_colours;

	CUDA_RNG* rngs;

	void send_scene(const std::vector<std::unique_ptr<Visible>>& scene);

	void create_rngs();

	void generate_rays();

	void shade_rays();

	void render_rays();

public:

	GPURayTracer(const int spp, const int max_bounce) : spp(spp), max_bounce(max_bounce) {}

	FrameBuffer* render(const int h, const int w, CUDAVisible** const scene, const int scene_size, const Camera& camera);

};

__global__ void colour_space(FrameBuffer* const fb);

__global__ void cuda_create_rngs(CUDA_RNG* const rngs, const uint32_t ray_count);

__global__ void cuda_gen_rays(Ray* rays, const uint32_t ray_count, const Camera* const cam, const FrameBuffer* const fb, CUDA_RNG* const rngs, const int spp);

__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const uint32_t ray_count, CUDAVisible** const scene, const int scene_size, const int max_bounce, CUDA_RNG* const rngs);

__device__ Intersection* nearest_intersection(const Ray& ray, CUDAVisible** const scene, const int scene_size, const float tmin, const float tmax);

__device__ vec3 draw_sky(const Ray& ray);

__global__ void cuda_render_rays(vec3* ray_colours, const uint32_t ray_count, FrameBuffer* const fb, const int spp);


