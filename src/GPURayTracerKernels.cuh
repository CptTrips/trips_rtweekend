#pragma once

#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "FrameBuffer.cuh"
#include "Camera.cuh"
#include "ray.cuh"
#include "Intersection.h"
#include "visibles/CUDASphere.cuh"

#include "rand.h"
#include "UnifiedArray.cuh"

__global__ void cuda_create_rngs(UnifiedArray<CUDA_RNG>* const rngs);

__global__ void cuda_gen_rays(Ray* rays, const uint64_t rayCount, const uint64_t ray_offset_index, const uint64_t raysPerBatch, const Camera* const cam, const FrameBuffer* const fb, CUDA_RNG* const rngs, const uint64_t spp);

__global__ void cuda_render_rays(const int pixel_start_idx, const int pixel_end_idx, UnifiedArray<Ray>* p_rayBuffer, FrameBuffer* const fb, const uint64_t spp);

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

__device__ ImagePoint subPixel(const uint64_t rayID, const FrameBuffer* const fb, const uint64_t spp, CUDA_RNG& rng);

/*
__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const uint64_t rayCount, const uint64_t raysPerBatch, uint64_t ray_offset_index, const UnifiedArray<CUDAVisible*>* const scene, const int maxBounce, const float minFreePath, CUDA_RNG* const rngs);

__device__ Intersection* nearest_intersection(const Ray& ray, const UnifiedArray<CUDAVisible*>* const scene, const float tmin, const float tmax);
*/