#include "CUDAIntersectionEngine.cuh"


void CUDAIntersectionEngine::run(UnifiedArray<Ray>* rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* vertexBuffer, UnifiedArray<uint32_t>* indexBuffer, UnifiedArray<CUDASphere>* sphereBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer)
{

	uint32_t threads = MAX_THREAD_COUNT;
	uint32_t blocks = ((rayBuffer->size() - 1) / threads) + 1;

	find_intersections <<<blocks, threads >>> (rayBuffer, p_activeRayIndices, vertexBuffer, indexBuffer, sphereBuffer, p_triangleIntersectionBuffer, p_sphereIntersectionBuffer, minFreePath);

	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void find_intersections(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_vertexBuffer, UnifiedArray<uint32_t>* p_indexBuffer, UnifiedArray<CUDASphere>* p_sphereBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer, const float minFreePath)
{

	if (THREAD_ID >= p_activeRayIndices->size())
		return;
	
	Ray ray = (*p_rayBuffer)[(*p_activeRayIndices)[THREAD_ID]];

	Intersection triIntersection = find_triangle_intersections(ray, p_vertexBuffer, p_indexBuffer, minFreePath);

	Intersection sphereIntersection = find_sphere_intersections(ray, p_sphereBuffer, minFreePath);

	if (
		(isfinite(triIntersection.t) && triIntersection.normal.length() == 0)
		|| (isfinite(sphereIntersection.t) && sphereIntersection.normal.length() == 0)
	)
		printf("Bad intersection %d", THREAD_ID);

	(*p_triangleIntersectionBuffer)[THREAD_ID] = triIntersection;
	(*p_sphereIntersectionBuffer)[THREAD_ID] = sphereIntersection;
}

__device__ Intersection find_triangle_intersections(const Ray& ray, UnifiedArray<vec3>* p_vertexBuffer, UnifiedArray<uint32_t>* p_indexBuffer, const float minFreePath)
{

	uint32_t triangleCount = p_indexBuffer->size() / 3;

	uint32_t i, j, k; // Vertex indices

	Intersection ixn;
	Intersection tempIxn;

	for (uint32_t triangleID = 0; triangleID < triangleCount; triangleID++)
	{

		i = (*p_indexBuffer)[triangleID * 3];
		j = (*p_indexBuffer)[triangleID * 3 + 1];
		k = (*p_indexBuffer)[triangleID * 3 + 2];

		vec3 points[] = { (*p_vertexBuffer)[i], (*p_vertexBuffer)[j], (*p_vertexBuffer)[k] };

		Triangle tri(points, nullptr);

		tempIxn = tri.intersect(ray, minFreePath, INFINITY);

		if (tempIxn.t < ixn.t)
		{

			ixn = tempIxn;

			ixn.id = triangleID;
		}
	}

	return ixn;
}

__device__ Intersection find_sphere_intersections(const Ray& ray, UnifiedArray<CUDASphere>* p_sphereBuffer, const float minFreePath)
{

	Intersection ixn, tempIxn;

	for (uint32_t sphereID = 0; sphereID < p_sphereBuffer->size(); sphereID++)
	{

		CUDASphere tempSphere = (*p_sphereBuffer)[sphereID];

		tempIxn = tempSphere.intersect(ray, minFreePath, INFINITY);

		if (tempIxn.t < ixn.t)
		{

			ixn = tempIxn;

			ixn.id = sphereID;
		}
	}

	return ixn;
}
