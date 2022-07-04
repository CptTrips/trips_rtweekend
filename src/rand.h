#ifndef RAND_H
#define RAND_H

#include "vec3.cuh"
#include <curand_kernel.h>
#include <random>

#define CPU_SEED 1

class CPU_RNG {
  public:
    CPU_RNG();
    float sample();
    vec3 sample_uniform_sphere();
};


class CUDA_RNG {

    curandState r;
public:
    __device__ CUDA_RNG(const int seed, const int seq);
    __device__ float sample();
    //__device__ ~CUDA_RNG();
    __device__ vec3 sample_uniform_sphere();
};

#endif
