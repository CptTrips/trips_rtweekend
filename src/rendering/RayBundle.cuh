#pragma once

#include "geometry/ray.cuh"

#include "memory/UnifiedArray.cuh"


struct RayBundle
{

	UnifiedArray<Ray>* p_rayArray;

	// Struct which keeps track of which rays
	UnifiedArray<uint32_t>* p_activeRayIndices;
};