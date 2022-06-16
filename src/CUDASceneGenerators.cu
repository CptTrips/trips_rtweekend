#include "CUDASceneGenerators.cuh"


void teardown_scene(CUDAScene* scene)
{

	cuda_teardown_scene << <1, 1 >> > (scene);

	checkCudaErrors(cudaFree(scene));

	checkCudaErrors(cudaDeviceSynchronize());

}


__global__ void cuda_teardown_scene(CUDAScene* scene)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id == 0)
	{
		printf("Visibles %u, Materials %u", scene->visibles->size(), scene->materials->size());
		scene->~CUDAScene();
	}
}

CUDAScene* random_balls(const int ball_count)
{

	CUDAScene* scenery;

    CUDAScene* host_scene = new CUDAScene();

	checkCudaErrors(cudaMalloc(&scenery, sizeof(CUDAScene)));

	checkCudaErrors(cudaMemcpy(scenery, host_scene, sizeof(CUDAScene), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize());

	create_scene_buffers << <1, 1 >> > (scenery, ball_count, ball_count);

	checkCudaErrors(cudaDeviceSynchronize());

	int threads = 512;

	int blocks = ball_count / threads + 1;

	gen_random_balls << <blocks, threads >> > (scenery, ball_count);

	checkCudaErrors(cudaDeviceSynchronize());

	return scenery;
}


__global__ void create_scene_buffers(CUDAScene* scenery, const int visible_count, const int material_count)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id == 0)
	{
		Array<CUDAVisible*>* visibles = new Array<CUDAVisible*>(visible_count);

		Array<Material<CUDA_RNG>*>* materials = new Array<Material<CUDA_RNG>*>(visible_count);

		scenery->set_visibles(visibles);

		scenery->set_materials(materials);
	}
}

__global__ void gen_random_balls(CUDAScene* const scene, const int ball_count)
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

		(*scene->visibles)[id] = new CUDASphere(center, r, m);
		(*scene->materials)[id] = m;

	}

}
/*

CUDAScene* single_ball()
{
	Array<CUDAVisible*>* visibles = new Array<CUDAVisible*>(1);
	Array<Material<CUDA_RNG>*>* materials = new Array<Material<CUDA_RNG>*>(1);

	gen_single_ball << <1, 1>> > (visibles, materials);

	cudaDeviceSynchronize();

    CUDAScene* scenery = new CUDAScene(visibles, materials);

	return scenery;
}

__global__ void gen_single_ball(Array<CUDAVisible*>* visibles, Array<Material<CUDA_RNG>*>* materials)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < 1)
	{
		vec3 center = vec3(3.f, 0.f, 0.f);
		float radius = 1.f;
		Material<CUDA_RNG>* mat = new Diffuse<CUDA_RNG>(vec3(1.f, 0.f, 0.f));
		(*visibles)[id] = new CUDASphere(center, radius, mat);
		(*materials)[id] = mat;
	}
}



CUDAScene* single_triangle()
{

	CUDAScene* scenery;

	cudaMalloc(&scenery, sizeof(CUDAScene));

	gen_single_triangle << <1, 1 >> > (scenery);

	cudaDeviceSynchronize();

	return scenery;
}


__global__ void gen_single_triangle(CUDAScene* const scenery)
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

CUDAScene* single_cube()
{

	CUDAScene* scenery;

	checkCudaErrors(cudaMalloc(&scenery, sizeof(CUDAScene)));

	gen_single_cube << <1, 1 >> > (scenery);

	checkCudaErrors(cudaDeviceSynchronize());

	return scenery;
}

__global__ void gen_single_cube(CUDAScene* const scenery)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id == 0)
	{

		vec3 translation = vec3(0.f, 0.f, 0.f);

		Array<vec3>* vertex_array = new Array<vec3>(8);

		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				for (int k = 0; k < 2; k++)
				{
					(*vertex_array)[i + 2 * j + 4 * k] = vec3(i, j, k);
				}

		Array<uint32_t>* index_array = new Array<uint32_t>(36);

		int indices[36] = {
			0, 2, 1, 2, 3, 1,
			1, 5, 4, 0, 1, 4,
			4, 5, 7, 4, 7, 6,
			6, 7, 3, 3, 2, 6,
			1, 3, 5, 1, 7, 5,
			0, 4, 6, 0, 6, 2
		};

		for (int i = 0; i < 36; i++)
			(*index_array)[i] = indices[i];


		Material<CUDA_RNG>* mat = new Diffuse<CUDA_RNG>(vec3(0.7f, 0.1f, 0.2f));

		scenery[id] = new Mesh(vertex_array, index_array, mat);
	}
}
*/
