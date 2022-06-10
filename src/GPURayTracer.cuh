#pragma once
#include "device_launch_parameters.h"
#include "visibles/sphere.cuh"
#include "FrameBuffer.cuh"
#include <vector>
#include <algorithm>
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
	const uint32_t max_rays_per_batch = 22 * 1e6;

	const uint32_t rays_per_batch;

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

	void allocate_rays();

	void generate_rays(const uint64_t ray_offset_index);

	void shade_rays(const uint64_t ray_offset_index);

	void render_rays(const uint64_t ray_offset_index);

public:

	GPURayTracer(const int spp, const int max_bounce) : spp(spp), max_bounce(max_bounce), rays_per_batch(spp * (max_rays_per_batch / spp)) {}

	FrameBuffer* render(const int h, const int w, CUDAVisible** const scene, const int scene_size, const Camera& camera);

};

__global__ void colour_space(FrameBuffer* const fb);

__global__ void cuda_create_rngs(CUDA_RNG* const rngs, const uint32_t ray_count);

__global__ void cuda_gen_rays(Ray* rays, const uint64_t ray_count, const uint64_t ray_offset_index, const uint64_t rays_per_batch, const Camera* const cam, const FrameBuffer* const fb, CUDA_RNG* const rngs, const int spp);

__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const uint64_t ray_count, const uint64_t rays_per_batch, uint64_t ray_offset_index, CUDAVisible** const scene, const int scene_size, const int max_bounce, CUDA_RNG* const rngs);

__device__ Intersection* nearest_intersection(const Ray& ray, CUDAVisible** const scene, const int scene_size, const float tmin, const float tmax);

__device__ vec3 draw_sky(const Ray& ray);

__global__ void cuda_render_rays(const int pixel_start_idx, const int pixel_end_idx, vec3* ray_colours, FrameBuffer* const fb, const int spp);


