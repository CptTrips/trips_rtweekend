#pragma once

#include "memory/UnifiedArray.cuh"

#include "visibles/CUDASphere.cuh"

#include "maths/vec3.cuh"

#include "rendering/Mesh.cuh"

#include <memory>
#include "memory/ManagedPtr.h"

struct Scene
{

	std::shared_ptr<MeshOwner> m_mesh;

	std::shared_ptr<UnifiedArray<CUDASphere>> m_sphereArray;
	std::shared_ptr<UnifiedArray<vec3>> m_sphereColourArray;

	/*
	Scene()
		: m_mesh(nullptr)
		, m_sphereArray(nullptr)
		, m_sphereColourArray(nullptr)
	{}
	*/

	Scene(uint64_t vertexCount, uint64_t triangleCount, uint64_t sphereCount);
};
