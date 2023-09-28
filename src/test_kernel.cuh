#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include "memory/Array.cuh"
#include "memory/UnifiedArray.cuh"

__global__ void output_int_array(UnifiedArray<Array<uint32_t>*>* int_arrays);

void test_nested_array();
