#pragma once

#include "UnifiedArray.cuh"
#include "visibles/CUDASphere.cuh"
#include "vec3.cuh"

#include <memory>
#include "ManagedPtr.h"

struct Scene
{

	std::shared_ptr<UnifiedArray<vec3>> m_vertexBuffer;
	std::shared_ptr<UnifiedArray<uint32_t>> m_indexBuffer;
	std::shared_ptr<UnifiedArray<vec3>> m_triColourBuffer;

	std::shared_ptr<UnifiedArray<CUDASphere>> m_sphereBuffer;
	std::shared_ptr<UnifiedArray<vec3>> m_sphereColourBuffer;

	Scene()
		: m_vertexBuffer(nullptr)
		, m_indexBuffer(nullptr)
		, m_triColourBuffer(nullptr)
		, m_sphereBuffer(nullptr)
		, m_sphereColourBuffer(nullptr)
	{}

};

