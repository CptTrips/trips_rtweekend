#include "rendering/CUDAIntersectionEngine.cuh"


void CUDAIntersectionEngine::run(UnifiedArray<Ray>* rayArray, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* vertexArray, UnifiedArray<uint32_t>* indexArray, UnifiedArray<CUDASphere>* sphereArray, UnifiedArray<Intersection>* p_triangleIntersectionArray, UnifiedArray<Intersection>* p_sphereIntersectionArray)
{

	uint32_t threads = MAX_THREAD_COUNT;
	uint32_t blocks = ((rayArray->size() - 1) / threads) + 1;

	find_intersections <<<blocks, threads >>> (rayArray, p_activeRayIndices, vertexArray, indexArray, sphereArray, p_triangleIntersectionArray, p_sphereIntersectionArray, minFreePath);

	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void find_intersections(UnifiedArray<Ray>* p_rayArray, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_vertexArray, UnifiedArray<uint32_t>* p_indexArray, UnifiedArray<CUDASphere>* p_sphereArray, UnifiedArray<Intersection>* p_triangleIntersectionArray, UnifiedArray<Intersection>* p_sphereIntersectionArray, const float minFreePath)
{

	if (THREAD_ID >= p_activeRayIndices->size())
		return;
	
	Ray ray = (*p_rayArray)[(*p_activeRayIndices)[THREAD_ID]];

	Intersection triIntersection = find_triangle_intersections(ray, p_vertexArray, p_indexArray, minFreePath);

	Intersection sphereIntersection = find_sphere_intersections(ray, p_sphereArray, minFreePath);

	/*
	if (
		(isfinite(triIntersection.t) && triIntersection.normal.length() == 0)
		|| (isfinite(sphereIntersection.t) && sphereIntersection.normal.length() == 0)
	)
		printf("Bad intersection %d", THREAD_ID);
		*/

	(*p_triangleIntersectionArray)[THREAD_ID] = triIntersection;
	(*p_sphereIntersectionArray)[THREAD_ID] = sphereIntersection;
}

__device__ Intersection find_triangle_intersections(const Ray& ray, UnifiedArray<vec3>* p_vertexArray, UnifiedArray<uint32_t>* p_indexArray, const float minFreePath)
{

	uint32_t triangleCount = p_indexArray->size() / 3;

	uint32_t i, j, k; // Vertex indices

	Intersection ixn;
	Intersection tempIxn;

	for (uint32_t triangleID = 0; triangleID < triangleCount; triangleID++)
	{

		i = (*p_indexArray)[triangleID * 3];
		j = (*p_indexArray)[triangleID * 3 + 1];
		k = (*p_indexArray)[triangleID * 3 + 2];

		vec3 points[] = { (*p_vertexArray)[i], (*p_vertexArray)[j], (*p_vertexArray)[k] };

		Triangle tri(points, nullptr);

		tempIxn = tri.intersect(ray, minFreePath, INFINITY);

		tempIxn.id = triangleID;

		ixn = (tempIxn.t < ixn.t) ? tempIxn : ixn;
	}

	return ixn;
}

__device__ Intersection find_sphere_intersections(const Ray& ray, UnifiedArray<CUDASphere>* p_sphereArray, const float minFreePath)
{

	Intersection ixn, tempIxn;

	for (uint32_t sphereID = 0; sphereID < p_sphereArray->size(); sphereID++)
	{

		CUDASphere tempSphere = (*p_sphereArray)[sphereID];

		tempIxn = tempSphere.intersect(ray, minFreePath, INFINITY);

		tempIxn.id = sphereID;

		ixn = (tempIxn.t < ixn.t) ? tempIxn : ixn;
	}

	return ixn;
}
