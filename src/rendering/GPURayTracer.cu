#include "rendering/GPURayTracer.cuh"

#include "rendering/GPURayTracerKernels.cuh"

#define DEBUG false

using std::shared_ptr;
using std::cout;
using std::endl;

void GPURayTracer::terminateRays(UnifiedArray<Ray>* p_rayArray, UnifiedArray<uint32_t>* p_activeRayIndices)
{

	uint32_t threads = max_threads;
	uint32_t blocks = p_activeRayIndices->size() / threads + 1;

	cuda_terminate_rays << <blocks, threads >> > (p_rayArray, p_activeRayIndices);

	checkCudaErrors(cudaDeviceSynchronize());
}

GPURayTracer::GPURayTracer(RayTracerConfig config)
	: xRes(config.xRes)
	, yRes(config.yRes)
	, spp(config.spp)
	, maxBounce(config.maxBounce)
	, rayCount(spp * xRes * yRes)
	, raysPerBatch(std::min(rayCount, spp * (maxRaysPerBatch / spp)))
	, ixnEngine(std::make_unique<BranchingTriangleIntersector>(config.minFreePath), config.minFreePath) 
{

	showDeviceProperties();

	increaseStackLimit();
}

shared_ptr<FrameBuffer> GPURayTracer::render(const Scene& scene, const Camera& camera)
{

	// Allocate Frame Array
	shared_ptr<FrameBuffer> m_fb { make_managed<FrameBuffer>(yRes, xRes) };

	// Make a copy of the camera in managed memory
	m_cam =  make_managed<Camera>(camera);

	// Allocate ray data (Ray, colour, rng)
	allocate_rays();

	// RNG for each ray
	create_rngs();

	// Package these in an IntersectionArray
	shared_ptr<UnifiedArray<Intersection>> m_triangleIntersectionArray = make_managed<UnifiedArray<Intersection>>(m_rayArray->size());
	shared_ptr<UnifiedArray<Intersection>> m_sphereIntersectionArray = make_managed<UnifiedArray<Intersection>>(m_rayArray->size());

	auto m_mesh{ scene.m_mesh->getFinder() };

	auto m_rayBundle{ make_managed<RayBundle>(RayBundle{m_rayArray.get(), nullptr, m_triangleIntersectionArray.get(), m_sphereIntersectionArray.get()}) };

	using milli = std::chrono::milliseconds;

	auto start = std::chrono::high_resolution_clock::now();

	for (uint64_t rayIDOffset = 0; rayIDOffset < rayCount; rayIDOffset += raysPerBatch)
	{

		std::cout << "Ray progress " << (float)rayIDOffset / (float)rayCount * 100 << "% " << rayIDOffset << " / " << rayCount << std::endl;

		generatePrimaryRays(rayIDOffset, m_fb.get());

		shared_ptr<UnifiedArray<uint32_t>> m_activeRayIndices{ resetActiveRays(m_rayArray->size()) };


		for (uint16_t bounce = 0; bounce < maxBounce; bounce++)
		{

			cout << endl << "  Bounce " << bounce << endl;

			m_rayBundle->p_activeRayIndices = m_activeRayIndices.get();

			ixnEngine.run(m_rayBundle.get(), m_mesh.get(), scene.m_sphereArray.get());

			colourRays(m_rayArray.get(), m_activeRayIndices.get(), m_mesh->p_triangleColourArray, scene.m_sphereColourArray.get(), m_triangleIntersectionArray.get(), m_sphereIntersectionArray.get());

			m_activeRayIndices = gatherActiveRays(m_activeRayIndices.get(), m_triangleIntersectionArray.get(), m_sphereIntersectionArray.get());

			cout << "  " << m_activeRayIndices->size() << " rays still active" << endl;

			if (m_activeRayIndices->size() == 0)
				break;

			scatterRays(m_rayArray.get(), m_activeRayIndices.get(), m_mesh->p_vertexArray, m_mesh->p_indexArray, scene.m_sphereArray.get(), m_triangleIntersectionArray.get(), m_sphereIntersectionArray.get());

			cout << endl;
		}

		terminateRays(m_rayArray.get(), m_activeRayIndices.get());

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
	m_rayArray = make_managed<UnifiedArray<Ray>>(raysPerBatch);


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

	cuda_gen_rays<<<blocks, threads>>>(m_rayArray->data(), rayCount, raysPerBatch, ray_offset_index, m_cam.get(), m_fb, m_rngs->data(), spp);

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


void GPURayTracer::colourRays(UnifiedArray<Ray>* p_rayArray, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_triangleColurArray, UnifiedArray<vec3>* p_sphereColourArray, UnifiedArray<Intersection>* p_triangleIntersectionArray, UnifiedArray<Intersection>* p_sphereIntersectionArray)
{

	uint32_t threads = max_threads;
	uint32_t blocks = raysPerBatch / threads + 1;

	cuda_colour_rays << <blocks, threads >> > (p_rayArray, p_activeRayIndices, p_triangleColurArray, p_sphereColourArray, p_triangleIntersectionArray, p_sphereIntersectionArray);

	checkCudaErrors(cudaDeviceSynchronize());
}

void GPURayTracer::renderRays(const uint64_t ray_offset_index, FrameBuffer* m_fb)
{

	uint32_t threads = max_threads;

	uint64_t pixel_start_idx = ray_offset_index / (uint64_t)spp;

	uint64_t pixel_end_idx = pixel_start_idx + raysPerBatch / spp; // not including this index

	pixel_end_idx = std::min(pixel_end_idx, static_cast<uint64_t>(m_fb->h * m_fb->w));

	uint64_t pixel_batch_size = pixel_end_idx - pixel_start_idx;

	uint64_t blocks = pixel_batch_size / threads + 1;

	std::cout << "render_rays blocks: " << blocks << ", threads: " << threads << std::endl;

	cuda_render_rays << <blocks, threads >> > (pixel_start_idx, pixel_end_idx, m_rayArray.get(), m_fb, spp);

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

std::shared_ptr<UnifiedArray<uint32_t>> GPURayTracer::gatherActiveRays(UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<Intersection>* p_triangleIntersectionArray, UnifiedArray<Intersection>* p_sphereIntersectionArray)
{

	uint32_t length = p_activeRayIndices->size();

	// Create 0, 1 mask for intersections
	UnifiedArray<uint32_t>* p_mask = new UnifiedArray<uint32_t>(length);

	// Create scan of mask
	UnifiedArray<uint32_t>* p_scan = new UnifiedArray<uint32_t>(length);

	KernelLaunchParams klp(max_threads);

	cuda_is_active<<<klp.blocks(length), klp.maxThreads>>>(p_mask, p_triangleIntersectionArray, p_sphereIntersectionArray);
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

void GPURayTracer::scatterRays(UnifiedArray<Ray>* p_rayArray, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_vertexArray, UnifiedArray<uint32_t>* p_indexArray, UnifiedArray<CUDASphere>* p_sphereArray, UnifiedArray<Intersection>* p_triangleIntersectionArray, UnifiedArray<Intersection>* p_sphereIntersectionArray)
{

	KernelLaunchParams klp(max_threads);

	cuda_scatter_rays << <klp.blocks(p_activeRayIndices->size()), klp.maxThreads >> > (
		p_rayArray,
		p_activeRayIndices,
		p_vertexArray,
		p_indexArray,
		p_sphereArray,
		p_triangleIntersectionArray,
		p_sphereIntersectionArray,
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
