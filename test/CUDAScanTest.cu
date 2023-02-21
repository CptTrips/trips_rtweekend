#include "CUDAScanTest.cuh"

const uint32_t ARRAY_SIZE = 1 << 20;

const uint32_t THREADS = 512;

TestContext::TestContext(const uint32_t size) : p_in(new UnifiedArray<uint32_t>(size)), p_out(new UnifiedArray<uint32_t>(size))
{
}

TestContext::~TestContext()
{

	delete p_in;
	delete p_out;
}


TEST_CASE("exp2i")
{

	SUBCASE("i = 0")
	{

		CHECK(exp2i(0) == 1);
	}

	SUBCASE("i > 0")
	{
		uint32_t j = 1;

		for (uint32_t i = 1; i < 30; i++)
		{
			CHECK(exp2i(i) == j * 2);

			j *= 2;
		}
	}
}

TEST_CASE("log2i")
{

	SUBCASE("i = 1")
	{

		CHECK(log2i(1) == 0);
	}

	SUBCASE("i = 3")
	{

		CHECK(log2i(3) == 1);
	}

	SUBCASE("i = 2^j")
	{

		uint32_t i = 2;

		for (uint32_t j = 1; j < 30; j++)
		{

			CHECK(log2i(i) == j);

			i *= 2;
		}
	}
}

TEST_CASE("Trivial Array (positive length, all zeros)")
{

	TestContext tc(ARRAY_SIZE);

	for (uint32_t i = 0; i < ARRAY_SIZE; i++)
		(*tc.p_in)[i] = 0;

	cudaScan(tc.p_in, tc.p_out);

	checkCudaErrors(cudaDeviceSynchronize());

	SUBCASE("Same length")
	{

		CHECK(tc.p_out->size() == tc.p_in->size());
	}

	SUBCASE("Correct output")
	{
		for (uint32_t i = 0; i < ARRAY_SIZE; i++)
			CHECK((*tc.p_out)[i] == 0);
	}
}


TEST_CASE("All ones")
{

	TestContext tc(ARRAY_SIZE);

	for (uint32_t i = 0; i < ARRAY_SIZE; i++)
		(*tc.p_in)[i] = 1;

	cudaScan(tc.p_in, tc.p_out);

	checkCudaErrors(cudaDeviceSynchronize());

	SUBCASE("Correct output")
	{

		for (uint32_t i = 0; i < ARRAY_SIZE; i++)
			CHECK((*tc.p_out)[i] == i + 1);
	}
}
