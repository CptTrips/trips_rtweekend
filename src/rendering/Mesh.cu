#include "rendering/Mesh.cuh"

#include <iostream>
#include <exception>

#include "memory/ManagedPtr.h"

MeshOwner::MeshOwner(uint64_t vertexCount, uint64_t triangleCount)
	: p_vertexArray(make_managed<UnifiedArray<vec3>>(vertexCount))
	, p_indexArray(make_managed<UnifiedArray<uint32_t>>(triangleCount * 3))
	, p_faceNormalArray(make_managed<UnifiedArray<vec3>>(triangleCount))
	, p_triangleColourArray(make_managed<UnifiedArray<vec3>>(triangleCount))
{

}

void MeshOwner::calculateFaceNormals()
{

	if ((p_faceNormalArray->size() * 3) != p_indexArray->size())
		throw std::runtime_error("Bad size for face normal array");

	std::cout << "Calculating face normals..." << std::endl;

	for (uint64_t i = 0; i < p_faceNormalArray->size(); i++)
	{

		vec3
			a{(*p_vertexArray)[(*p_indexArray)[3 * i + 0]]},
			b{(*p_vertexArray)[(*p_indexArray)[3 * i + 1]]},
			c{(*p_vertexArray)[(*p_indexArray)[3 * i + 2]]};

		(*p_faceNormalArray)[i] = cross(b - a, c - b).normalise();
	}
}

std::shared_ptr<MeshFinder> MeshOwner::getFinder() const
{

	return make_managed<MeshFinder>(MeshFinder{
		p_vertexArray.get(),
		p_indexArray.get(),
		p_faceNormalArray.get(),
		p_triangleColourArray.get()}
	);
}


