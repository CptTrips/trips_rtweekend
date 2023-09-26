#pragma once

#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "FrameBuffer.cuh"
#include "Camera.cuh"
#include "ray.cuh"
#include "Error.cuh"
#include "rand.h"
#include "UnifiedArray.cuh"
#include "CUDAIntersectionEngine.cuh"
#include "CUDAScan.cuh"
#include "KernelLaunchParams.h"
#include "RayTracerConfig.h"

#include "Scene.cuh"

#include <chrono>
#include <vector>
#include <algorithm>
#include <memory>

#define THREAD_ID threadIdx.x + blockIdx.x * blockDim.x

using std::shared_ptr;
using std::make_shared;
using std::cout;
using std::endl;


class GPURayTracer
{

	unsigned int xRes, yRes;

	const int max_threads = 512;

	// Maximum number of rays to dispatch at once (GPU memory limited)
	static constexpr uint32_t max_rays_per_batch = 22 * 1e6;

	uint32_t rays_per_batch = 0;

	int spp = 0;

	int maxBounce = 0;

	uint64_t rayCount = 0;

	FrameBuffer* h_fb = nullptr;

	Camera* d_cam = nullptr;

	UnifiedArray<Ray>* p_rayBuffer = nullptr;

	CUDA_RNG* rngs = nullptr;

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

	void terminateRays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices);

public:

	GPURayTracer(RayTracerConfig config);

	FrameBuffer* render(const Scene& scene, const Camera& camera);

};

__global__ void cuda_create_rngs(CUDA_RNG* const rngs, const uint32_t rayCount);

__global__ void cuda_gen_rays(Ray* rays, const uint64_t rayCount, const uint64_t ray_offset_index, const uint64_t rays_per_batch, const Camera* const cam, const FrameBuffer* const fb, CUDA_RNG* const rngs, const int spp);

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

__global__ void cuda_terminate_rays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activateRayIndices);

__global__ void cuda_is_active(UnifiedArray<uint32_t>* p_mask, UnifiedArray<Intersection>* p_triIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer);

__device__ ImagePoint subPixel(const uint64_t rayID, const FrameBuffer* const fb, const int spp, CUDA_RNG& rng);

/*
__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const uint64_t rayCount, const uint64_t rays_per_batch, uint64_t ray_offset_index, const UnifiedArray<CUDAVisible*>* const scene, const int maxBounce, const float minFreePath, CUDA_RNG* const rngs);

__device__ Intersection* nearest_intersection(const Ray& ray, const UnifiedArray<CUDAVisible*>* const scene, const float tmin, const float tmax);
*/
