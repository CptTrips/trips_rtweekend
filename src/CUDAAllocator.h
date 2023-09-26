#pragma once

#include <memory>

#include <cuda_runtime.h>

template<class T>
class CUDAAllocator
{

public:
	typedef T value_type;

	static constexpr size_t sizeOfT = sizeof(T);


	T* allocate(std::size_t n)
	{

		if (n > std::numeric_limits<std::size_t>::max() / sizeOfT)
			throw std::bad_array_new_length();

		T* p;

		if (cudaSuccess == cudaMallocManaged(static_cast<void**>(&p), sizeOfT * n))
		{

			cudaDeviceSynchronize();

			return p;
		}

		throw std::bad_alloc();
	}

	void deallocate(T* p, std::size_t n) noexcept
	{

		cudaDeviceSynchronize();

		if (cudaSuccess != cudaFree(static_cast<void*>(p)))
			std::cerr << "CUDAAllocator deallocation failed" << std::endl;
	}

};

template<class T, class U>
bool operator==(const CUDAAllocator <T>&, const CUDAAllocator <U>&) { return true; }
 
template<class T, class U>
bool operator!=(const CUDAAllocator <T>&, const CUDAAllocator <U>&) { return false; }
