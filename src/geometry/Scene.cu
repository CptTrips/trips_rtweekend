#include "geometry/Scene.cuh"


Scene::Scene(uint64_t vertexCount, uint64_t triangleCount, uint64_t sphereCount)
	: m_mesh(make_managed<MeshOwner>(vertexCount, triangleCount))
{

}
