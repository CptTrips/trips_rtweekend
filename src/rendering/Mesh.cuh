#pragma once

#include "memory/UnifiedArray.cuh"

#include "maths/vec3.cuh"

struct Mesh
{

	UnifiedArray<vec3>* p_vertexArray;
	UnifiedArray<uint32_t> p_indexArray;
	UnifiedArray<vec3>* p_triangleColourArray;
};