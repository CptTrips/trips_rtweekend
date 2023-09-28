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

	void run(UnifiedArray<Ray>* rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* vertexBuffer, UnifiedArray<uint32_t>* indexBuffer, UnifiedArray<CUDASphere>* sphereBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer);
};

__global__ void find_intersections(UnifiedArray<Ray>* p_rayBuffer, UnifiedArray<uint32_t>* p_activeRayIndices, UnifiedArray<vec3>* p_vertexBuffer, UnifiedArray<uint32_t>* p_indexBuffer, UnifiedArray<CUDASphere>* p_sphereBuffer, UnifiedArray<Intersection>* p_triangleIntersectionBuffer, UnifiedArray<Intersection>* p_sphereIntersectionBuffer, const float minFreePath);

__device__ Intersection find_triangle_intersections(const Ray& ray, UnifiedArray<vec3>* p_vertexBuffer, UnifiedArray<uint32_t>* p_indexBuffer, const float minFreePath);

__device__ Intersection find_sphere_intersections(const Ray& ray, UnifiedArray<CUDASphere>* p_sphereBuffer, const float minFreePath);

