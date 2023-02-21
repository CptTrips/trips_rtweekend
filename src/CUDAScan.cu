#include "CUDAScan.cuh"

__host__ __device__ uint32_t log2i(uint32_t i)
{

	uint32_t out = 0;

	if (i == 0)
		printf("Error: log2(0)\n");

	i = i >> 1;

	while (i != 0)
	{

		i = i >> 1;

		out++;
	}

	return out;
}

__host__ __device__ uint32_t exp2i(uint32_t i)
{

	uint32_t out = 1;

	while (i != 0)
	{

		out = out << 1;
		i--;
	}

	return out;
}

__global__ void hills_steele_step(UnifiedArray<uint32_t>* p_in, UnifiedArray<uint32_t>* p_out, uint32_t offset)
{

	if (THREAD_ID < p_out->size())
		(*p_out)[THREAD_ID] = (THREAD_ID < offset) ? (*p_in)[THREAD_ID] : (*p_in)[THREAD_ID] + (*p_in)[THREAD_ID - offset];
}

__global__ void copy(UnifiedArray<uint32_t>* p_in, UnifiedArray<uint32_t>* p_out)
{

	if (THREAD_ID < p_in->size())
		(*p_out)[THREAD_ID] = (*p_in)[THREAD_ID];
}

void swapPointers(UnifiedArray<uint32_t>*& p_front, UnifiedArray<uint32_t>*& p_back)
{

	UnifiedArray<uint32_t>* p_temp = p_back;

	p_back = p_front;

	p_front = p_temp;
}


void cudaScan(UnifiedArray<uint32_t>* p_in, UnifiedArray<uint32_t>* p_out)
{

	UnifiedArray<uint32_t>* p_front = p_out; // This is the wrong way round but we will swap them immediately
	UnifiedArray<uint32_t>* p_back = new UnifiedArray<uint32_t>(p_in->size());

	KernelLaunchParams klp(CUDA_SCAN_THREADS, p_in->size());
	copy << <klp.blocks, klp.threads >> > (p_in, p_back);
	checkCudaErrors(cudaDeviceSynchronize());

	for (uint32_t i = 0; i <= log2i(p_in->size()); i++)
	{

		if (i != 0)
			swapPointers(p_front, p_back);

		KernelLaunchParams klp(CUDA_SCAN_THREADS, p_in->size());
		hills_steele_step<<<klp.blocks, klp.threads>>>(p_back, p_front, exp2i(i));
		checkCudaErrors(cudaDeviceSynchronize());
	}

	if (p_out == p_back)
	{

		KernelLaunchParams klp(CUDA_SCAN_THREADS, p_front->size());
		copy << <klp.blocks, klp.threads >> > (p_front, p_out);
		checkCudaErrors(cudaDeviceSynchronize());

		swapPointers(p_front, p_back);
	}

	delete p_back;
}

