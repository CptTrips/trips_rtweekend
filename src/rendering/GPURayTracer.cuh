#pragma once


#include "memory/ManagedPtr.h"

#include "rendering/FrameBuffer.cuh"
#include "geometry/Camera.cuh"
#include "geometry/ray.cuh"
#include "utility/Error.cuh"
#include "maths/rand.cuh"
#include "memory/UnifiedArray.cuh"
#include "rendering/CUDAIntersectionEngine.cuh"
#include "algorithms/CUDAScan.cuh"
#include "utility/KernelLaunchParams.h"
#include "utility/RayTracerConfig.h"

#include "geometry/Scene.cuh"

#include <chrono>
#include <vector>
#include <algorithm>
#include <memory>

struct IntersectionArray
{

	std::shared_ptr<UnifiedArray<Intersection>> m_triangleIntersectionArray, m_sphereIntersectionArray;
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

	std::shared_ptr<UnifiedArray<Ray>> m_rayArray;

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

	void colourRays(UnifiedArray<Ray>* p_rayArray, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_triangleColurArray, UnifiedArray<vec3>* p_sphereColourArray, UnifiedArray<Intersection>* p_triangleIntersectionArray, UnifiedArray<Intersection>* p_sphereIntersectionArray);

	void renderRays(const uint64_t ray_offset_index, FrameBuffer* m_fb);

	std::shared_ptr<UnifiedArray<uint32_t>> resetActiveRays(const uint32_t& bufferSize);

	std::shared_ptr<UnifiedArray<uint32_t>> gatherActiveRays(UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<Intersection>* p_triangleIntersectionArray, UnifiedArray<Intersection>* p_sphereIntersectionArray);

	void scatterRays(UnifiedArray<Ray>* p_rayArray, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_vertexArray, UnifiedArray<uint32_t>* p_indexArray, UnifiedArray<CUDASphere>* p_sphereArray, UnifiedArray<Intersection>* p_triangleIntersectionArray, UnifiedArray<Intersection>* p_sphereIntersectionArray);

	/*
	void shade_rays(const uint64_t ray_offset_index);
	*/

	void terminateRays(UnifiedArray<Ray>* p_rayArray, UnifiedArray<uint32_t>* p_activeRayIndices);

public:

	GPURayTracer(RayTracerConfig config);

	std::shared_ptr<FrameBuffer> render(const Scene& scene, const Camera& camera);

};

