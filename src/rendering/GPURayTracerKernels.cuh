#pragma once

#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "rendering/FrameBuffer.cuh"
#include "geometry/Camera.cuh"
#include "geometry/ray.cuh"
#include "geometry/Intersection.h"
#include "visibles/CUDASphere.cuh"

#include "maths/rand.cuh"
#include "memory/UnifiedArray.cuh"

__global__ void cuda_create_rngs(UnifiedArray<CUDA_RNG>* const rngs);

__global__ void cuda_gen_rays(UnifiedArray<Ray>* p_rays, const uint64_t rayCount, const uint64_t ray_offset_index, const Camera* const cam, const FrameBuffer* const fb, CUDA_RNG* const rngs, const uint64_t spp);

__global__ void cuda_render_rays(const uint64_t pixel_start_idx, const uint64_t pixel_end_idx, UnifiedArray<Ray>* p_rayArray, FrameBuffer* const fb, const uint64_t spp);

__device__ vec3 gamma_correction(const vec3& col_in);

__host__ __device__ vec3 draw_sky(const Ray& ray);

__global__ void cuda_reset_active_rays(UnifiedArray<uint32_t>* p_activeRayIndices);

__global__ void cuda_colour_rays(UnifiedArray<Ray>* p_rayArray, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_triangleColourArray, UnifiedArray<vec3>* p_sphereColourArray, UnifiedArray<Intersection>* p_triangleIntersectionArray, UnifiedArray<Intersection>* p_sphereIntersectionArray);

__global__ void cuda_scatter_rays(
	UnifiedArray<Ray>* p_rayArray,
	UnifiedArray<uint32_t>* p_activeRayIndices,
	UnifiedArray<vec3>* p_vertexArray,
	UnifiedArray<uint32_t>* p_indexArray,
	UnifiedArray<CUDASphere>* p_sphereArray,
	UnifiedArray<Intersection>* p_triangleIntersectionArray,
	UnifiedArray<Intersection>* p_sphereIntersectionArray,
	//UnifiedArray<Material<CUDA_RNG>>* p_materialArray,
	CUDA_RNG* const rngs
);

__global__ void cuda_terminate_rays(UnifiedArray<Ray>* p_rayArray, UnifiedArray<uint32_t>* p_activateRayIndices);

__global__ void cuda_is_active(UnifiedArray<uint32_t>* p_mask, UnifiedArray<Intersection>* p_triIntersectionArray, UnifiedArray<Intersection>* p_sphereIntersectionArray);

__device__ ImagePoint subPixel(const uint64_t rayID, const FrameBuffer* const fb, const uint64_t spp, CUDA_RNG& rng);

/*
__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const uint64_t rayCount, const uint64_t raysPerBatch, uint64_t ray_offset_index, const UnifiedArray<CUDAVisible*>* const scene, const int maxBounce, const float minFreePath, CUDA_RNG* const rngs);

__device__ Intersection* nearest_intersection(const Ray& ray, const UnifiedArray<CUDAVisible*>* const scene, const float tmin, const float tmax);
*/