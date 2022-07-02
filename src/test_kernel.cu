#include "test_kernel.cuh"

class int_array_holder
{
public:
	__device__ __host__ int_array_holder(Array<uint32_t>* p_int_array): p_int_array(p_int_array) {}

	Array<uint32_t>* p_int_array;
};

__global__ void read_int_array(UnifiedArray<int_array_holder*>* p_iah_h)
{
	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id < (*p_iah_h)[0]->p_int_array->size())
		printf("%d ", (*(*p_iah_h)[0]->p_int_array)[id]);
}


__global__ void write_int_array(UnifiedArray<Array<uint32_t>*>* int_arrays, UnifiedArray<int_array_holder*>* p_iah_h)
{

	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id == 0)
	{
		(*p_iah_h)[0] = new int_array_holder((*int_arrays)[0]);
	}

}

__global__ void output_int_array(UnifiedArray<Array<uint32_t>*>* int_arrays)
{
	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id < (*int_arrays)[0]->size())
	{
		Array<uint32_t>* p_int_array = (*int_arrays)[0];
		printf("%d ", (*p_int_array)[id]);
	}
}

void test_nested_array()
{
	unsigned int n = 1024;

	Array<uint32_t>* int_array = new Array<uint32_t>(n);

	UnifiedArray<Array<uint32_t>*>* int_arrays = new UnifiedArray<Array<uint32_t>*>(1);

	UnifiedArray<int_array_holder*>* p_iah_h = new UnifiedArray<int_array_holder*>(1);

	for (unsigned int i = 0; i < n; i++)
	{
		(*int_array)[i] = i;
	}

	(*int_arrays)[0] = int_array->to_device();

	checkCudaErrors(cudaDeviceSynchronize());

	delete int_array;

	// Launch kernel which outputs contents of int_array
	output_int_array<<<4,256>>>(int_arrays);

	checkCudaErrors(cudaDeviceSynchronize());

	// Kernel which puts int_arrays[0] into a heap variable
	write_int_array<<<1,1>>>(int_arrays, p_iah_h);

	checkCudaErrors(cudaDeviceSynchronize());

	// Kernel which outputs contents of int_array via heap variable
	read_int_array<<<4,256>>>(p_iah_h);

	checkCudaErrors(cudaDeviceSynchronize());

	delete int_arrays;

}
