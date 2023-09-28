#include "rendering/GPURayTracer.cuh"

#include "rendering/GPURayTracerKernels.cuh"

#define DEBUG false

using std::shared_ptr;
using std::cout;
using std::endl;

void GPURayTracer::terminateRays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices)
{

	uint32_t threads = max_threads;
	uint32_t blocks = p_activeRayIndices->size() / threads + 1;

	cuda_terminate_rays << <blocks, threads >> > (p_rayBuffer, p_activeRayIndices);

	checkCudaErrors(cudaDeviceSynchronize());
}

GPURayTracer::GPURayTracer(RayTracerConfig config)
	: xRes(config.xRes)
	, yRes(config.yRes)
	, spp(config.spp)
	, maxBounce(config.maxBounce)
	, rayCount(spp * xRes * yRes)
	, raysPerBatch(std::min(rayCount, spp * (maxRaysPerBatch / spp)))
	, ixnEngine(config.minFreePath)
{

	showDeviceProperties();

	increaseStackLimit();
}

shared_ptr<FrameBuffer> GPURayTracer::render(const Scene& scene, const Camera& camera)
{

	// Allocate Frame Buffer
	shared_ptr<FrameBuffer> m_fb { make_managed<FrameBuffer>(yRes, xRes) };

	// Make a copy of the camera in managed memory
	m_cam =  make_managed<Camera>(camera);

	// Allocate ray data (Ray, colour, rng)
	allocate_rays();

	// RNG for each ray
	create_rngs();

	// Package these in an IntersectionBuffer
	shared_ptr<UnifiedArray<Intersection>> m_triangleIntersectionBuffer = make_managed<UnifiedArray<Intersection>>(m_rayBuffer->size());
	shared_ptr<UnifiedArray<Intersection>> m_sphereIntersectionBuffer = make_managed<UnifiedArray<Intersection>>(m_rayBuffer->size());

	using milli = std::chrono::milliseconds;

	auto start = std::chrono::high_resolution_clock::now();

	for (uint64_t rayIDOffset = 0; rayIDOffset < rayCount; rayIDOffset += raysPerBatch)
	{

		std::cout << "Ray progress " << (float)rayIDOffset / (float)rayCount * 100 << "% " << rayIDOffset << " / " << rayCount << std::endl;

		generatePrimaryRays(rayIDOffset, m_fb.get());

		shared_ptr<UnifiedArray<uint32_t>> m_activeRayIndices{ resetActiveRays(m_rayBuffer->size()) };

		for (uint16_t bounce = 0; bounce < maxBounce; bounce++)
		{

			cout << endl << "  Bounce " << bounce << endl;

			ixnEngine.run(m_rayBuffer.get(), m_activeRayIndices.get(), scene.m_vertexBuffer.get(), scene.m_indexBuffer.get(), scene.m_sphereBuffer.get(), m_triangleIntersectionBuffer.get(), m_sphereIntersectionBuffer.get());

			colourRays(m_rayBuffer.get(), m_activeRayIndices.get(), scene.m_triColourBuffer.get(), scene.m_sphereColourBuffer.get(), m_triangleIntersectionBuffer.get(), m_sphereIntersectionBuffer.get());

			m_activeRayIndices = gatherActiveRays(m_activeRayIndices.get(), m_triangleIntersectionBuffer.get(), m_sphereIntersectionBuffer.get());

			cout << "  " << m_activeRayIndices->size() << " rays still active" << endl;

			if (m_activeRayIndices->size() == 0)
				break;

			scatterRays(m_rayBuffer.get(), m_activeRayIndices.get(), scene.m_vertexBuffer.get(), scene.m_indexBuffer.get(), scene.m_sphereBuffer.get(), m_triangleIntersectionBuffer.get(), m_sphereIntersectionBuffer.get());

			cout << endl;
		}

		terminateRays(m_rayBuffer.get(), m_activeRayIndices.get());

		renderRays(rayIDOffset, m_fb.get());

		std::cout << std::endl;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	auto finish = std::chrono::high_resolution_clock::now();
	
	std::cout << "Frametime: "
		<< std::chrono::duration_cast<milli>(finish - start).count() << "ms"
		<< std::endl;

	return m_fb;
}

void GPURayTracer::allocate_rays()
{

	// Allocate rays
	m_rayBuffer = make_managed<UnifiedArray<Ray>>(raysPerBatch);


	checkCudaErrors(cudaDeviceSynchronize());

}

void GPURayTracer::create_rngs()
{

	m_rngs = make_managed<UnifiedArray<CUDA_RNG>>(raysPerBatch);

	cudaDeviceSynchronize();

	uint32_t threads = max_threads;
	uint32_t blocks = raysPerBatch / threads + 1;

	cuda_create_rngs << <blocks, threads >> > (m_rngs.get());

	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());
}

void GPURayTracer::generatePrimaryRays(const uint64_t ray_offset_index, const FrameBuffer* const m_fb)
{

	uint32_t threads = max_threads;
	uint32_t blocks = raysPerBatch / threads + 1;

	std::cout << "generatePrimaryRays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_gen_rays<<<blocks, threads>>>(m_rayBuffer->data(), rayCount, raysPerBatch, ray_offset_index, m_cam.get(), m_fb, m_rngs->data(), spp);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaGetLastError());

}

void GPURayTracer::showDeviceProperties()
{

	cout << "Device Properties" << endl;

	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0); // assume one CUDA device

	cout << "Max Grid Size: " << prop.maxGridSize[0] << "x " << prop.maxGridSize[1] << "y " << prop.maxGridSize[2] << "z " << endl;
	cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << endl;
	cout << "Shared Mem Per Block: " << prop.sharedMemPerBlock << endl;

	cout << endl;
}

void GPURayTracer::increaseStackLimit()
{

	size_t stack_limit;

	checkCudaErrors(cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize));

	std::cout << "Default stack limit: " << stack_limit << std::endl;

	checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 8*stack_limit));

	checkCudaErrors(cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize));

	std::cout << "New stack limit: " << stack_limit << std::endl;

	cout << endl;
}


void GPURayTracer::colourRays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_triangleColurBuffer, UnifiedArray<vec3>* p_sphereColourBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer)
{

	uint32_t threads = max_threads;
	uint32_t blocks = raysPerBatch / threads + 1;

	cuda_colour_rays << <blocks, threads >> > (p_rayBuffer, p_activeRayIndices, p_triangleColurBuffer, p_sphereColourBuffer, p_triangleIntersectionBuffer, p_sphereIntersectionBuffer);

	checkCudaErrors(cudaDeviceSynchronize());
}

void GPURayTracer::renderRays(const uint64_t ray_offset_index, FrameBuffer* m_fb)
{

	uint32_t threads = max_threads;

	int pixel_start_idx = ray_offset_index / (uint64_t)spp;

	int pixel_end_idx = pixel_start_idx + raysPerBatch / spp; // not including this index

	pixel_end_idx = std::min(pixel_end_idx, m_fb->h * m_fb->w);

	int pixel_batch_size = pixel_end_idx - pixel_start_idx;

	int blocks = pixel_batch_size / threads + 1;

	std::cout << "render_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_render_rays << <blocks, threads >> > (pixel_start_idx, pixel_end_idx, m_rayBuffer.get(), m_fb, spp);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaGetLastError());
}

shared_ptr<UnifiedArray<uint32_t>> GPURayTracer::resetActiveRays(const uint32_t& bufferSize)
{

	shared_ptr<UnifiedArray<uint32_t>> m_activeRayIndices{ make_managed<UnifiedArray<uint32_t>>(bufferSize) };

	uint32_t threads = max_threads;
	uint32_t blocks = raysPerBatch / threads + 1;

	cuda_reset_active_rays << <blocks, threads >> > (m_activeRayIndices.get());

	checkCudaErrors(cudaDeviceSynchronize());

	return m_activeRayIndices;
}

std::shared_ptr<UnifiedArray<uint32_t>> GPURayTracer::gatherActiveRays(UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer)
{

	uint32_t length = p_activeRayIndices->size();

	// Create 0, 1 mask for intersections
	UnifiedArray<uint32_t>* p_mask = new UnifiedArray<uint32_t>(length);

	// Create scan of mask
	UnifiedArray<uint32_t>* p_scan = new UnifiedArray<uint32_t>(length);

	KernelLaunchParams klp(max_threads, length);

	cuda_is_active<<<klp.blocks, klp.threads>>>(p_mask, p_triangleIntersectionBuffer, p_sphereIntersectionBuffer);
	checkCudaErrors(cudaDeviceSynchronize());

	cudaScan(p_mask, p_scan);
	checkCudaErrors(cudaDeviceSynchronize());

	uint32_t activeRayCount = (*p_scan)[length - 1];

	if (activeRayCount > length)
		throw std::runtime_error("Scan failed: more active rays than rays\n");

	// Create new active ray index array
	shared_ptr<UnifiedArray<uint32_t>> m_newActiveRayIndices{ make_managed<UnifiedArray<uint32_t>>(activeRayCount) };

	for (uint32_t i = 0; i < length; i++)
	{

		if ((*p_mask)[i] == 1)
			(*m_newActiveRayIndices)[(*p_scan)[i] - 1] = (*p_activeRayIndices)[i];
	}

	delete p_mask;
	delete p_scan;
	
	return m_newActiveRayIndices;
}

void GPURayTracer::scatterRays(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_vertexBuffer, UnifiedArray<uint32_t>* p_indexBuffer, UnifiedArray<CUDASphere>* p_sphereBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer)
{

	KernelLaunchParams klp(max_threads, p_activeRayIndices->size());

	cuda_scatter_rays << <klp.blocks, klp.threads >> > (
		p_rayBuffer,
		p_activeRayIndices,
		p_vertexBuffer,
		p_indexBuffer,
		p_sphereBuffer,
		p_triangleIntersectionBuffer,
		p_sphereIntersectionBuffer,
		m_rngs->data()
	);

	checkCudaErrors(cudaDeviceSynchronize());
}


/*

void GPURayTracer::shade_rays(const uint64_t ray_offset_index)
{
	
	uint32_t threads = 512;

	uint32_t blocks = raysPerBatch / threads + 1;

	std::cout << "shade_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	size_t stack_size;

	checkCudaErrors(cudaThreadGetLimit(&stack_size, cudaLimitStackSize));

	cuda_shade_ray << <blocks, threads >> > (rays, ray_colours, rayCount, raysPerBatch, ray_offset_index, visibles, maxBounce, minFreePath, rngs);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaGetLastError());
}


*/
