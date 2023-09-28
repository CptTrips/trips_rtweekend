#pragma once

#include <cstdint>

#define THREAD_ID threadIdx.x + blockIdx.x * blockDim.x

struct KernelLaunchParams
{

private:
	static constexpr uint32_t DEFAULT_MAX_THREADS = 512;

public:
	KernelLaunchParams(const uint32_t maxThreads=DEFAULT_MAX_THREADS)
		: maxThreads(maxThreads)
	{
		
	}

	uint32_t blocks(const uint32_t work) { return work / maxThreads + 1; }
	

	uint32_t maxThreads;
};
