#pragma once

#include "device_launch_parameters.h"

#include "rendering/RayBundle.cuh"
#include "rendering/Mesh.cuh"

#include "geometry/Intersection.h"

#include "visibles/Triangle.cuh"

#include "utility/KernelLaunchParams.h"


class TriangleIntersector
{

protected:
	float minFreePath;

	KernelLaunchParams klp{};

public:
	TriangleIntersector(float minFreePath=1e-3) : minFreePath(minFreePath) {};

	virtual void findTriangleIntersections(RayBundle* const m_rayBundle, const MeshFinder* const m_mesh) = 0;

};


class BranchingTriangleIntersector : public TriangleIntersector
{

public:
	BranchingTriangleIntersector(float minFreePath=1e-3);

	virtual void findTriangleIntersections(RayBundle* const m_rayBundle, const MeshFinder* const m_mesh) override;
};

__global__ void find_triangle_intersections(RayBundle* const m_rayBundle, const MeshFinder* const m_mesh, const float minFreePath);


class BranchlessTriangleIntersector : public TriangleIntersector
{

public:
	BranchlessTriangleIntersector(float minFreePath = 1e-3);

	virtual void findTriangleIntersections(RayBundle* const m_rayBundle, const MeshFinder* const m_mesh) override;
};


__host__ __device__ Triangle getTriangle(const MeshFinder* const m_mesh, uint64_t triangleID);
