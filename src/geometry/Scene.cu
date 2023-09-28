#include "geometry/Scene.cuh"

std::shared_ptr<Mesh> Scene::getManagedMesh() const
{

	return make_managed<Mesh>(Mesh{ m_vertexArray.get(), m_indexArray.get(), m_triColourArray.get() });
}
