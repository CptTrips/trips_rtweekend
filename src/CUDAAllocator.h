#pragma once

#include <memory>

#include <iostream>

#include <cuda_runtime.h>

#include "Error.cuh"

template<class T>
class CUDAAllocator
{

public:
	typedef T value_type;

	static constexpr size_t sizeOfT = sizeof(T);

	CUDAAllocator() = default;

	template<typename U>
	constexpr CUDAAllocator(const CUDAAllocator<U>& other) noexcept {}

	template<typename U>
	struct rebind
	{

		typedef CUDAAllocator<U> other;
	};

	T* allocate(std::size_t n)
	{

		if (n > std::numeric_limits<std::size_t>::max() / sizeOfT)
			throw std::bad_array_new_length();

		T* p;

		if (cudaSuccess == cudaMallocManaged(&p, sizeOfT * n))
		{

			checkCudaErrors(cudaDeviceSynchronize());

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
