#pragma once

#include "UnifiedArray.cuh"
#include "visibles/CUDASphere.cuh"
#include "vec3.cuh"

struct Scene
{

	UnifiedArray<vec3>* p_vertexBuffer;
	UnifiedArray<uint32_t>* p_indexBuffer;
	UnifiedArray<vec3>* p_triColourBuffer;

	UnifiedArray<CUDASphere>* p_sphereBuffer;
	UnifiedArray<vec3>* p_sphereColourBuffer;

	Scene()
		: p_vertexBuffer(nullptr)
		, p_indexBuffer(nullptr)
		, p_triColourBuffer(nullptr)
		, p_sphereBuffer(nullptr)
		, p_sphereColourBuffer(nullptr)
	{}

};

