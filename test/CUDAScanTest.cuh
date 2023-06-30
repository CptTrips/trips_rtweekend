#include "doctest.h"

#include <device_launch_parameters.h>
#include "curand_kernel.h"

#include "CUDAScan.cuh"
#include "UnifiedArray.cuh"
#include "KernelLaunchParams.h"


struct TestContext
{

	TestContext(const uint32_t size);

	TestContext(const TestContext& other) = delete;
	TestContext& operator=(const TestContext& other) = delete;

	TestContext(TestContext&& other) = delete;
	TestContext& operator=(TestContext&& other) = delete;

	~TestContext();

	UnifiedArray<uint32_t>* p_in;
	UnifiedArray<uint32_t>* p_out;
};
