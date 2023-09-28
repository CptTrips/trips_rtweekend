#pragma once


#include "ManagedPtr.h"

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

struct IntersectionBuffer
{

	std::shared_ptr<UnifiedArray<Intersection>> m_triangleIntersectionBuffer, m_sphereIntersectionBuffer;
};


class GPURayTracer
{

	// Maximum number of rays to dispatch at once (GPU memory limited)
	static constexpr uint64_t maxRaysPerBatch = 22 * 1e6;

	static constexpr uint32_t max_threads = 512;

	unsigned int xRes, yRes;

	uint64_t spp;

	uint32_t maxBounce;

	uint64_t rayCount;

	uint64_t raysPerBatch;

	std::shared_ptr<UnifiedArray<Ray>> m_rayBuffer;

	std::shared_ptr<UnifiedArray<CUDA_RNG>> m_rngs;

	std::shared_ptr<Camera> m_cam;

	CUDAIntersectionEngine ixnEngine;

	//void send_scene(const std::vector<std::unique_ptr<Visible>>& scene);

	void allocateRenderResources();

	void create_rngs();

	void allocate_rays();

	void generatePrimaryRays(const uint64_t ray_offset_index, const FrameBuffer* const m_fb);

	void showDeviceProperties();

	void increaseStackLimit();

	void colourRays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_triangleColurBuffer, UnifiedArray<vec3>* p_sphereColourBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer);

	void renderRays(const uint64_t ray_offset_index, FrameBuffer* m_fb);

	std::shared_ptr<UnifiedArray<uint32_t>> resetActiveRays(const uint32_t& bufferSize);

	std::shared_ptr<UnifiedArray<uint32_t>> gatherActiveRays(UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer);

	void scatterRays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_vertexBuffer, UnifiedArray<uint32_t>* p_indexBuffer, UnifiedArray<CUDASphere>* p_sphereBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer);

	/*
	void shade_rays(const uint64_t ray_offset_index);
	*/

	void terminateRays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices);

public:

	GPURayTracer(RayTracerConfig config);

	std::shared_ptr<FrameBuffer> render(const Scene& scene, const Camera& camera);

};

