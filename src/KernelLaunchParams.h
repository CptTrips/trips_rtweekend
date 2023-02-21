#pragma once

#include <cstdint>

struct KernelLaunchParams
{

	KernelLaunchParams(const uint32_t threads, const uint32_t work)
		: threads(threads), work(work), blocks(work / threads + 1)
	{
		
	}

	uint32_t threads, work, blocks;
};
