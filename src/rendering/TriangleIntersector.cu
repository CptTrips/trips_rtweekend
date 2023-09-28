#include "TriangleIntersector.cuh"

BranchingTriangleIntersector::BranchingTriangleIntersector(float minFreePath)
	: TriangleIntersector(minFreePath)
{
}

void BranchingTriangleIntersector::findTriangleIntersections(RayBundle* const m_rayBundle, const Mesh* const m_mesh)
{

	checkCudaErrors(cudaDeviceSynchronize());

	uint32_t blocks = klp.blocks(m_rayBundle->p_rayArray->size() - 1);

	find_triangle_intersections <<<blocks, klp.maxThreads >>> (m_rayBundle, m_mesh, minFreePath);

	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void find_triangle_intersections(RayBundle* const m_rayBundle, const Mesh* const m_mesh, const float minFreePath)
{

	if (THREAD_ID >= m_rayBundle->p_activeRayIndices->size())
		return;

	auto* const p_rayArray = m_rayBundle->p_rayArray;
	auto* const p_activeRayIndices = m_rayBundle->p_activeRayIndices;
	auto* const p_triangleIntersectionArray = m_rayBundle->p_triangleIntersectionArray;

	auto* const p_indexArray = m_mesh->p_indexArray;
	auto* const p_vertexArray = m_mesh->p_vertexArray;
	
	Ray ray = (*p_rayArray)[(*p_activeRayIndices)[THREAD_ID]];

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

	(*p_triangleIntersectionArray)[THREAD_ID] = ixn;
}

