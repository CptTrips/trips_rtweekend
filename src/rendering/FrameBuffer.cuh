#pragma once
#include "maths/vec3.cuh"
#include "memory/Managed.cuh"
#include "utility/Error.cuh"

class FrameBuffer : public Managed
{


public:

	int h, w;

	char* buffer;

	FrameBuffer(const int& h, const int& w);

	FrameBuffer(const FrameBuffer& fb);

	FrameBuffer& operator=(const FrameBuffer& fb);

	~FrameBuffer();

	__host__ __device__ void set_pixel(const int& r, const int& c, const vec3& col);
};
