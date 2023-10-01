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

#include "rendering/TriangleIntersector.cuh"

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

	std::shared_ptr<UnifiedArray<CUDA_RNG>> m_rngs;

	std::shared_ptr<Camera> m_cam;

	CUDAIntersectionEngine ixnEngine;

	//void send_scene(const std::vector<std::unique_ptr<Visible>>& scene);

	void createRNGs();

	void generatePrimaryRays(const RayBundle& rayBundle, const uint64_t ray_offset_index, const FrameBuffer* const m_fb);

	void showDeviceProperties();

	void increaseStackLimit();

	void colourRays(
		const RayBundle& rayBundle,
		const Scene& scene
	);

	void renderRays(const RayBundle& rayBundle, const uint64_t ray_offset_index, FrameBuffer* m_fb);

	std::shared_ptr<UnifiedArray<uint32_t>> resetActiveRays(const uint32_t& bufferSize);

	std::shared_ptr<UnifiedArray<uint32_t>> gatherActiveRays(const RayBundle& rayBundle);

	void scatterRays(const RayBundle& rayBundle, const Scene& scene);

	/*
	void shade_rays(const uint64_t ray_offset_index);
	*/

	void terminateRays(const RayBundle& rayBundle);

public:

	GPURayTracer(RayTracerConfig config);

	std::shared_ptr<FrameBuffer> render(const Scene& scene, const Camera& camera);

};

