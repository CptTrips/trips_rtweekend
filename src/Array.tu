
template<typename T>
__host__ __device__ Array<T>::Array()
{
	data = NULL;

	_size = 0;
}

template<typename T>
__host__ __device__ Array<T>::Array(const uint32_t size) : _size(size)
{

	data = new T[_size];

}

template<typename T>
__host__ __device__ Array<T>::Array(const Array<T>& b)
{

	copy(b);

}

template<typename T>
__host__ __device__ Array<T>& Array<T>::operator=(const Array<T>& b)
{
	if (this == &b)
		return *this;
	
	if (data)
		delete[] data;

	copy(b);


	return *this;
}


template<typename T>
__host__ __device__ void Array<T>::copy(const Array<T>& b)
{

	_size = b.size();

	data = new T[_size];

	for (int i = 0; i < _size; i++)
	{
		data[i] = b[i];
	}

}

template<typename T>
__host__ __device__ Array<T>::~Array()
{

	if (data)
		delete[] data;
	
}

template<typename T>
__host__ __device__ T& Array<T>::operator[](const uint32_t i)
{
	assert(i < _size);
	return data[i];
}

template<typename T>
__host__ __device__ const T& Array<T>::operator[](const uint32_t i) const
{
	assert(i < _size);
	return data[i];
}

template<typename T>
__host__ __device__ uint32_t Array<T>::size() const
{
	return _size;
}


// Copy to device
template<typename T>
__host__ Array<T>* Array<T>::to_device() const
{
	T* device_data;

	uint64_t bytes = _size * sizeof(T);

	cudaMalloc(&device_data, bytes);

	cudaMemcpy(device_data, data, bytes, cudaMemcpyHostToDevice);

	Array<T> device_array = Array<T>();

	device_array._size = _size;

	device_array.data = device_data;

	Array<T>* p_device_array;

	cudaMalloc(&p_device_array, sizeof(Array<T>));

	cudaMemcpy(p_device_array, &device_array, sizeof(Array<T>), cudaMemcpyHostToDevice);

	device_array.data = NULL;

	return p_device_array;
}

