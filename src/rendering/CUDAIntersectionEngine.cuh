#pragma once

#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "geometry/ray.cuh"
#include "geometry/Intersection.h"

#include "memory/UnifiedArray.cuh"

#include "visibles\Triangle.cuh"
#include "visibles\CUDASphere.cuh"

#include "utility/KernelLaunchParams.h"

#include "rendering/TriangleIntersector.cuh"
#include "rendering/RayBundle.cuh"
#include "rendering/Mesh.cuh"

#include <vector>
#include <algorithm>
#include <memory>


class CUDAIntersectionEngine
{

	KernelLaunchParams klp;

	std::unique_ptr<TriangleIntersector> p_triangleIntersector;

public:

	float minFreePath = 1e-3;

	CUDAIntersectionEngine(std::unique_ptr<TriangleIntersector> p_triangleIntersector, const float minFreePath=1e-3) : p_triangleIntersector(std::move(p_triangleIntersector)), minFreePath(minFreePath) {};

	void run(RayBundle* const m_rayBundle, const Mesh* const m_mesh, UnifiedArray<CUDASphere>* sphereArray);
};


__global__ void find_sphere_intersections(UnifiedArray<Ray>* p_rayArray, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<CUDASphere>* p_sphereArray, UnifiedArray<Intersection>* p_sphereIntersectionArray, const float minFreePath);

