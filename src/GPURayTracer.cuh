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
#include "UnifiedArray.cuh"

struct GPURenderProperties
{
	int h;
	int w;
	int spp;
	int max_bounce;
};

class GPURayTracer
{

	const int max_threads = 512;

	// Maximum number of rays to dispatch at once (GPU memory limited)
	const uint32_t max_rays_per_batch = 22 * 1e6;

	uint32_t rays_per_batch = 0;

	int spp = 0;

	int max_bounce = 0;

	uint64_t ray_count = 0;

	UnifiedArray<CUDAVisible*>* visibles = NULL;

	FrameBuffer* h_fb = NULL;

	Camera* d_cam = NULL;

	Ray* rays = NULL;

	vec3* ray_colours = NULL;

	CUDA_RNG* rngs = NULL;

	void send_scene(const std::vector<std::unique_ptr<Visible>>& scene);

	void create_rngs();

	void allocate_rays();

	void generate_rays(const uint64_t ray_offset_index);

	void shade_rays(const uint64_t ray_offset_index);

	void render_rays(const uint64_t ray_offset_index);

public:

	FrameBuffer* render(const GPURenderProperties& render_properies, UnifiedArray<CUDAVisible*>* visibles, const Camera& camera);

};

__global__ void colour_space(FrameBuffer* const fb);

__global__ void cuda_create_rngs(CUDA_RNG* const rngs, const uint32_t ray_count);

__global__ void cuda_gen_rays(Ray* rays, const uint64_t ray_count, const uint64_t ray_offset_index, const uint64_t rays_per_batch, const Camera* const cam, const FrameBuffer* const fb, CUDA_RNG* const rngs, const int spp);

__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const uint64_t ray_count, const uint64_t rays_per_batch, uint64_t ray_offset_index, const UnifiedArray<CUDAVisible*>* const scene, const int max_bounce, CUDA_RNG* const rngs);

__device__ Intersection* nearest_intersection(const Ray& ray, const UnifiedArray<CUDAVisible*>* const scene, const float tmin, const float tmax);

__device__ vec3 draw_sky(const Ray& ray);

__global__ void cuda_render_rays(const int pixel_start_idx, const int pixel_end_idx, vec3* ray_colours, FrameBuffer* const fb, const int spp);

__device__ vec3 gamma_correction(const vec3& col_in);


