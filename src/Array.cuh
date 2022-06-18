#pragma once
#include "cuda_runtime.h"
#include "Managed.cuh"

template<typename T>
class Array
{
public:
	__host__ __device__ Array();
	__host__ __device__ Array(const uint32_t size);
	__host__ __device__ Array(const Array& b);
	__host__ __device__ Array& operator=(const Array& b);
	__host__ __device__ ~Array();
	__host__ __device__ T& operator[](const uint32_t i);
	__host__ __device__ const T& operator[](const uint32_t i) const;
	__host__ __device__ uint32_t size() const;

	__host__ __device__ const T* get_data() const;

	__host__ Array<T>* to_device() const;

private:

	__host__ __device__ void copy(const Array& b);

	T* data;
	uint32_t _size;
};

#include "Array.tu"
