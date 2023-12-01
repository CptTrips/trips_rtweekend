
#pragma once
#include "cuda_runtime.h"
#include "memory/Managed.cuh"
#include "utility/Error.cuh"

template<typename T>
class UnifiedArray : public Managed
{
public:
	__host__ UnifiedArray();
	__host__ UnifiedArray(const uint32_t size);

	__host__ UnifiedArray(const UnifiedArray& b) = delete;
	__host__ UnifiedArray& operator=(const UnifiedArray& b) = delete;

	__host__ ~UnifiedArray();
	__host__ __device__ T& operator[](const uint32_t i);
	__host__ __device__ const T& operator[](const uint32_t i) const;
	__host__ __device__ uint32_t size() const;
	__host__ __device__ T* data() const;

private:

	//__host__ void copy(const UnifiedArray& b);

	T* _data;
	uint32_t _size;
};


#include "memory/UnifiedArray.tu"