#include "CUDAScenes.cuh"


CUDAVisible** random_balls(const int ball_count)
{

    CUDAVisible** scenery;

	cudaMalloc(&scenery, ball_count * sizeof(scenery));

	int threads = 512;

	int blocks = ball_count / threads + 1;

	gen_random_balls << <blocks, threads >> > (scenery, ball_count);

	cudaDeviceSynchronize();

	return scenery;
}

__global__ void gen_random_balls(CUDAVisible** const scenery, const int ball_count)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;


	if (id < ball_count)
	{

		CUDA_RNG rng = CUDA_RNG(my_cuda_seed, id);

		float r = 0.33; // ball radius

		float xrange = 6.f;
		float yrange = 3.75;
		float zrange = 2.5f;

		float zoffset = -0.f;

		vec3 center = vec3(
			xrange * (2.f * rng.sample() - 1)
			,yrange * (2.f * rng.sample() - 1)
			,zoffset - zrange * rng.sample()
		);

		vec3 color = vec3(rng.sample(),rng.sample(),rng.sample());

		float roughness = 3.f*rng.sample();

		// Randomize the material
		Material<CUDA_RNG>* m;

		if (rng.sample() > .5f) {

			m = new Metal<CUDA_RNG>(color, roughness);

		} else {

			m = new Diffuse<CUDA_RNG>(color);

		}


		scenery[id] = new CUDASphere(center, r, m);
	}

}

CUDAVisible** single_ball()
{

    CUDAVisible** scenery;

	cudaMalloc(&scenery, sizeof(CUDAVisible*));

	gen_single_ball << <1, 1>> > (scenery);

	cudaDeviceSynchronize();

	return scenery;
}

__global__ void gen_single_ball(CUDAVisible** const scenery)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < 1)
	{
		vec3 center = vec3(3.f, 0.f, 0.f);
		float radius = 1.f;
		Material<CUDA_RNG>* mat = new Diffuse<CUDA_RNG>(vec3(1.f, 0.f, 0.f));
		scenery[id] = new CUDASphere(center, radius, mat);
	}
}



CUDAVisible** single_triangle()
{

	CUDAVisible** scenery;

	cudaMalloc(&scenery, sizeof(CUDAVisible*));

	gen_single_triangle << <1, 1 >> > (scenery);

	cudaDeviceSynchronize();

	return scenery;
}


__global__ void gen_single_triangle(CUDAVisible** const scenery)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id == 0)
	{
		vec3 a = vec3(0.f, 0.f, 1.f);

		vec3 b = vec3(0.f, 1.f, 1.f);

		vec3 c = vec3(1.f, 0.f, 1.f);

		vec3 points[3] = { a, b, c };

		Material<CUDA_RNG>* mat = new Metal<CUDA_RNG>(vec3(.5f, .2f, .2f), 0.1f);

		scenery[id] = new Triangle(points, mat);
	}
}
