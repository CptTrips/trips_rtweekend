#pragma once
#include "vec3.h"

class FrameBuffer
{


public:

	int h, w;

	char* buffer;

	FrameBuffer(const int& h, const int& w);

	FrameBuffer(const FrameBuffer& fb);

	FrameBuffer& operator=(const FrameBuffer& fb);

	~FrameBuffer();

	void set_pixel(const int& r, const int& c, const vec3& col);
};
