#pragma once

#include "device_launch_parameters.h"

#include "RayBundle.cuh"


class TriangleIntersector
{

public:

	virtual void findTriangleIntersections(const RayBundle* const m_rayBundle) = 0;

};