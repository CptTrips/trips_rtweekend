#pragma once

#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "FrameBuffer.cuh"
#include "Camera.h"
#include "ray.cuh"
#include "Error.cuh"
#include "rand.h"
#include "UnifiedArray.cuh"
#include "CUDAIntersectionEngine.cuh"

#include <chrono>
#include <vector>
#include <algorithm>
#include <memory>

#define THREAD_ID threadIdx.x + blockIdx.x * blockDim.x

using std::shared_ptr;
using std::make_shared;
using std::cout;
using std::endl;


struct GPURenderProperties
{
	int h;
	int w;
	int spp;
	int max_bounce;
	float min_free_path;
};



class GPURayTracer
{

	float min_free_path = 0;

	const int max_threads = 512;

	// Maximum number of rays to dispatch at once (GPU memory limited)
	const uint32_t max_rays_per_batch = 22 * 1e6;

	uint32_t rays_per_batch = 0;

	int spp = 0;

	int max_bounce = 0;

	uint64_t ray_count = 0;

	FrameBuffer* h_fb = NULL;

	Camera* d_cam = NULL;

	UnifiedArray<Ray>* p_rayBuffer = nullptr;

	CUDA_RNG* rngs = NULL;

	CUDAIntersectionEngine ixnEngine;

	//void send_scene(const std::vector<std::unique_ptr<Visible>>& scene);

	void create_rngs();

	void allocate_rays();

	void generate_primary_rays(const uint64_t ray_offset_index);

	void showDeviceProperties();

	void increaseStackLimit();

	void colourRays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_triangleColurBuffer, UnifiedArray<vec3>* p_sphereColourBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer);

	void render_rays(const uint64_t ray_offset_index);

	UnifiedArray<uint32_t>* resetActiveRays(const uint32_t& bufferSize);

	UnifiedArray<uint32_t>* gatherActiveRays(UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer);

	void scatterRays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_vertexBuffer, UnifiedArray<uint32_t>* p_indexBuffer, UnifiedArray<CUDASphere>* p_sphereBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer);

	/*
	void shade_rays(const uint64_t ray_offset_index);
	*/

public:

	GPURayTracer();

	FrameBuffer* render(const GPURenderProperties& render_properies, const Camera& camera);

};

__global__ void cuda_create_rngs(CUDA_RNG* const rngs, const uint32_t ray_count);

__global__ void cuda_gen_rays(Ray* rays, const uint64_t ray_count, const uint64_t ray_offset_index, const uint64_t rays_per_batch, const Camera* const cam, const FrameBuffer* const fb, CUDA_RNG* const rngs, const int spp);

__global__ void cuda_render_rays(const int pixel_start_idx, const int pixel_end_idx, UnifiedArray<Ray>* p_rayBuffer, FrameBuffer* const fb, const int spp);

__device__ vec3 gamma_correction(const vec3& col_in);

__host__ __device__ vec3 draw_sky(const Ray& ray);

__global__ void cuda_reset_active_rays(UnifiedArray<uint32_t>* p_activeRayIndices);

__global__ void cuda_colour_rays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_triangleColourBuffer, UnifiedArray<vec3>* p_sphereColourBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer);

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
);

__device__ vec3 scatter_ray(
	const Ray& rayIn,
	const vec3& normal,
	CUDA_RNG& rng
	//Material<CUDA_RNG>& material
);

__device__ vec3 specular_scatter(const Ray& ray, const vec3& normal, const float& roughness, CUDA_RNG& rng);

__device__ vec3 diffuse_scatter(const Ray& ray, const vec3& normal, CUDA_RNG& rng);
/*
__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const uint64_t ray_count, const uint64_t rays_per_batch, uint64_t ray_offset_index, const UnifiedArray<CUDAVisible*>* const scene, const int max_bounce, const float min_free_path, CUDA_RNG* const rngs);

__device__ Intersection* nearest_intersection(const Ray& ray, const UnifiedArray<CUDAVisible*>* const scene, const float tmin, const float tmax);
*/
