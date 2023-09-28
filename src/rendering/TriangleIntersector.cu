#include "TriangleIntersector.cuh"

BranchingTriangleIntersector::BranchingTriangleIntersector(float minFreePath)
	: TriangleIntersector(minFreePath)
{
}

void BranchingTriangleIntersector::findTriangleIntersections(RayBundle* const m_rayBundle, const MeshFinder* const m_mesh)
{

	checkCudaErrors(cudaDeviceSynchronize());

	uint32_t blocks = klp.blocks(m_rayBundle->p_rayArray->size() - 1);

	find_triangle_intersections <<<blocks, klp.maxThreads >>> (m_rayBundle, m_mesh, minFreePath);

	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void find_triangle_intersections(RayBundle* const m_rayBundle, const MeshFinder* const m_mesh, const float minFreePath)
{

	if (THREAD_ID >= m_rayBundle->p_activeRayIndices->size())
		return;

	const auto* const p_rayArray = m_rayBundle->p_rayArray;
	const auto* const p_activeRayIndices = m_rayBundle->p_activeRayIndices;
	auto* const p_triangleIntersectionArray = m_rayBundle->p_triangleIntersectionArray;

	const auto* const p_indexArray = m_mesh->p_indexArray;
	const auto* const p_vertexArray = m_mesh->p_vertexArray;
	const auto* const p_faceNormalArray = m_mesh->p_faceNormalArray;
	
	Ray ray = (*p_rayArray)[(*p_activeRayIndices)[THREAD_ID]];

	const uint32_t triangleCount = p_indexArray->size() / 3;

	uint32_t i, j, k; // Vertex indices

	Intersection ixn;
	Intersection tempIxn;

	for (uint32_t triangleID = 0; triangleID < triangleCount; triangleID++)
	{

		Triangle tri(getTriangle(m_mesh, triangleID));

		tempIxn = tri.intersect(ray, minFreePath, INFINITY);

		tempIxn.id = triangleID;

		ixn = (tempIxn.t < ixn.t) ? tempIxn : ixn;
	}

	(*p_triangleIntersectionArray)[THREAD_ID] = ixn;
}

__host__ __device__ Triangle getTriangle(const MeshFinder* const m_mesh, uint64_t triangleID)
{

	uint64_t i = (*m_mesh->p_indexArray)[triangleID * 3];
	uint64_t j = (*m_mesh->p_indexArray)[triangleID * 3 + 1];
	uint64_t k = (*m_mesh->p_indexArray)[triangleID * 3 + 2];

	vec3 vertices[] { (*m_mesh->p_vertexArray)[i], (*m_mesh->p_vertexArray)[j], (*m_mesh->p_vertexArray)[k] };


	return {vertices[0], vertices[1], vertices[2], (*m_mesh->p_faceNormalArray)[triangleID]};
}

BranchlessTriangleIntersector::BranchlessTriangleIntersector(float minFreePath)
	: TriangleIntersector(minFreePath)
{
}

void BranchlessTriangleIntersector::findTriangleIntersections(RayBundle* const m_rayBundle, const MeshFinder* const m_mesh)
{
}
