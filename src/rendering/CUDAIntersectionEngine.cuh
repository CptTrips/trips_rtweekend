#pragma once

#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "geometry/ray.cuh"
#include "memory/UnifiedArray.cuh"
#include "geometry/Intersection.h"
#include "visibles\Triangle.cuh"
#include "visibles\CUDASphere.cuh"

#include <vector>
#include <algorithm>

using std::vector;
using std::max;

static const uint16_t MAX_THREAD_COUNT = 512;
//static const uint16_t MAX_BLOCK_COUNT = 512;

#define THREAD_ID threadIdx.x + blockIdx.x * blockDim.x

class CUDAIntersectionEngine
{

public:

	float minFreePath = 1e-3;

	CUDAIntersectionEngine(const float& minFreePath=1e-3) : minFreePath(minFreePath) {};

	void run(UnifiedArray<Ray>* rayArray, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* vertexArray, UnifiedArray<uint32_t>* indexArray, UnifiedArray<CUDASphere>* sphereArray, UnifiedArray<Intersection>* p_triangleIntersectionArray, UnifiedArray<Intersection>* p_sphereIntersectionArray);
};

__global__ void find_intersections(UnifiedArray<Ray>* p_rayArray, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_vertexArray, UnifiedArray<uint32_t>* p_indexArray, UnifiedArray<CUDASphere>* p_sphereArray, UnifiedArray<Intersection>* p_triangleIntersectionArray, UnifiedArray<Intersection>* p_sphereIntersectionArray, const float minFreePath);

__device__ Intersection find_triangle_intersections(const Ray& ray, UnifiedArray<vec3>* p_vertexArray, UnifiedArray<uint32_t>* p_indexArray, const float minFreePath);

__device__ Intersection find_sphere_intersections(const Ray& ray, UnifiedArray<CUDASphere>* p_sphereArray, const float minFreePath);

