#include "rendering/CUDAIntersectionEngine.cuh"


void CUDAIntersectionEngine::run(RayBundle* const m_rayBundle, const MeshFinder* const m_mesh, UnifiedArray<CUDASphere>* sphereArray)
{

	checkCudaErrors(cudaDeviceSynchronize());

	find_sphere_intersections <<<klp.blocks(m_rayBundle->p_rayArray->size() - 1), klp.maxThreads >>> (m_rayBundle->p_rayArray, m_rayBundle->p_activeRayIndices, sphereArray, m_rayBundle->p_sphereIntersectionArray, minFreePath);

	p_triangleIntersector->findTriangleIntersections(m_rayBundle, m_mesh);
}


__global__ void find_sphere_intersections(UnifiedArray<Ray>* p_rayArray, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<CUDASphere>* p_sphereArray, UnifiedArray<Intersection>* p_sphereIntersectionArray, const float minFreePath)
{

	if (THREAD_ID >= p_activeRayIndices->size())
		return;
	
	Ray ray = (*p_rayArray)[(*p_activeRayIndices)[THREAD_ID]];

	Intersection ixn, tempIxn;

	for (uint32_t sphereID = 0; sphereID < p_sphereArray->size(); sphereID++)
	{

		CUDASphere tempSphere = (*p_sphereArray)[sphereID];

		tempIxn = tempSphere.intersect(ray, minFreePath, INFINITY);

		tempIxn.id = sphereID;

		ixn = (tempIxn.t < ixn.t) ? tempIxn : ixn;
	}

	/*
	if (isfinite(ixn.t) && ixn.normal.length() == 0)
		printf("Bad intersection %d", THREAD_ID);
	*/

	(*p_sphereIntersectionArray)[THREAD_ID] = ixn;
}
