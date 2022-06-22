template<typename T>
__host__ UnifiedArray<T>::UnifiedArray()
{
	data = NULL;

	_size = 0;
}

template<typename T>
__host__ UnifiedArray<T>::UnifiedArray(const uint32_t size) : _size(size)
{
	checkCudaErrors(cudaMallocManaged(&data, size * sizeof(T)));
	cudaDeviceSynchronize();
}


template<typename T>
__host__ UnifiedArray<T>::~UnifiedArray()
{
	cudaDeviceSynchronize();
	checkCudaErrors(cudaFree(data));
}

template<typename T>
__host__ __device__ T& UnifiedArray<T>::operator[](const uint32_t i)
{
	return data[i];
}

template<typename T>
__host__ __device__ const T& UnifiedArray<T>::operator[](const uint32_t i) const
{
	return data[i];
}

template<typename T>
__host__ __device__ uint32_t UnifiedArray<T>::size() const
{
	return _size;
}
