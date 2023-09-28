#include "maths/rand.cuh"

std::mt19937 gen;


std::uniform_real_distribution<float> rnd(0., 1.);


template<typename RNG_T>
vec3 sample_uniform_sphere(RNG_T& rng)
{

	vec3 p;

	do {

		p = 2.*vec3(rng.sample()-.5, rng.sample()-.5, rng.sample()-.5);

	} while (dot(p,p) > 1.);

	return p;

}

template __device__ vec3 sample_uniform_sphere<CUDA_RNG>(CUDA_RNG& rng);

template vec3 sample_uniform_sphere<CPU_RNG>(CPU_RNG& rng);


CPU_RNG::CPU_RNG(int seed) { gen.seed(seed); }

float CPU_RNG::sample(){

  return rnd(gen);
}

// Return a point sampled uniformly from sphere of radius 1
vec3 CPU_RNG::sample_uniform_sphere()
{

    return ::sample_uniform_sphere<CPU_RNG>(*this);
}



__device__ CUDA_RNG::CUDA_RNG(const int seed, const int seq)
{
    curand_init(seed, seq, 0, &r);

}

__device__ float CUDA_RNG::sample()
{
    return curand_uniform(&r);
}

/*
__device__ CUDA_RNG::~CUDA_RNG()
{
    delete r;
}
*/

__device__ vec3 CUDA_RNG::sample_uniform_sphere()
{

    return ::sample_uniform_sphere<CUDA_RNG>(*this);
}
