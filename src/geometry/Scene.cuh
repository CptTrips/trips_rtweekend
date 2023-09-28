#pragma once

#include "memory/UnifiedArray.cuh"

#include "visibles/CUDASphere.cuh"

#include "maths/vec3.cuh"

#include "rendering/Mesh.cuh"

#include <memory>
#include "memory/ManagedPtr.h"

struct Scene
{

	std::shared_ptr<UnifiedArray<vec3>> m_vertexArray;
	std::shared_ptr<UnifiedArray<uint32_t>> m_indexArray;
	std::shared_ptr<UnifiedArray<vec3>> m_triColourArray;

	std::shared_ptr<UnifiedArray<CUDASphere>> m_sphereArray;
	std::shared_ptr<UnifiedArray<vec3>> m_sphereColourArray;

	Scene()
		: m_vertexArray(nullptr)
		, m_indexArray(nullptr)
		, m_triColourArray(nullptr)
		, m_sphereArray(nullptr)
		, m_sphereColourArray(nullptr)
	{}

	/// <summary>
	/// Get a managed ptr to the mesh (triangle) parts of the scene
	/// </summary>
	std::shared_ptr<Mesh> getManagedMesh() const;

};

