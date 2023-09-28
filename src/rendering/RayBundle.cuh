#pragma once

#include "geometry/ray.cuh"
#include "geometry/Intersection.h"

#include "memory/UnifiedArray.cuh"

/// <summary>
/// Holds the data for rays as they bounce through the scene. SoA.
/// </summary>
struct RayBundle
{

	UnifiedArray<Ray>* p_rayArray;

	// Array which keeps track of which rays are still bouncing
	UnifiedArray<uint32_t>* p_activeRayIndices;

	// Array which stores closest triangle intersections for each active ray
	UnifiedArray<Intersection>* p_triangleIntersectionArray;

	// Array which stores closest sphere intersections for each active ray
	UnifiedArray<Intersection>* p_sphereIntersectionArray;

};