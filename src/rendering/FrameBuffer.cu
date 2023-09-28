#include "rendering/FrameBuffer.cuh"


FrameBuffer::FrameBuffer(const int& h, const int& w) : h(h), w(w)
{
	checkCudaErrors(cudaMallocManaged(&buffer, h * w * 3));
}

FrameBuffer::FrameBuffer(const FrameBuffer& fb)
{
	h = fb.h;
	w = fb.w;

	buffer = new char[h * w * 3];

	memcpy(buffer, fb.buffer, h * w * 3);
}

FrameBuffer& FrameBuffer::operator=(const FrameBuffer& fb)
{

	if (this == &fb)
		return *this;

	if (fb.h != h || fb.w != w)
	{
		delete[] buffer;

		h = fb.h;

		w = fb.w;

		buffer = new char[h * w * 3];

	}

	for (int r=0; r < h; r++)
		for (int c=0; c < w; c++)
			for (int k = 0; k < 3; k++)
			{
				int idx = r * w * 3 + c * 3 + k;

				buffer[idx] = fb.buffer[idx];
			}


	return *this;
}

FrameBuffer::~FrameBuffer()
{
	checkCudaErrors(cudaFree(buffer));

	checkCudaErrors(cudaDeviceSynchronize());
}

__host__ __device__ void FrameBuffer::set_pixel(const int& r, const int& c, const vec3& col)
{
	size_t pixel_index = r * w * 3 + c * 3;
	buffer[pixel_index] =     static_cast<uint8_t>(255.f*col.r());
	buffer[pixel_index + 1] = static_cast<uint8_t>(255.f*col.g());
	buffer[pixel_index + 2] = static_cast<uint8_t>(255.f*col.b());
}
