#pragma once
#include "vec3.cuh"
#include "Managed.cuh"

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
