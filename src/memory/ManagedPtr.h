#pragma once

#include <memory>

#include "memory/CUDAAllocator.h"

template<typename T, typename... Args>
std::shared_ptr<T> make_managed(Args&&... args)
{

	CUDAAllocator<T> alloc;

	return std::allocate_shared<T>(alloc, args...);
}