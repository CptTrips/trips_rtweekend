#include "CUDASceneGenerators.cuh"


CUDAScene* scene_factory(const int visible_count, const int material_count)
{

	CUDAScene* scene = new CUDAScene();

	Array<CUDAVisible*>* visibles = Array<CUDAVisible*>(visible_count).to_device();

	Array<Material<CUDA_RNG>*>* materials = Array<Material<CUDA_RNG>*>(visible_count).to_device();

	scene->visibles = visibles;

	scene->materials = materials;

	return scene;
}


CUDAScene* random_balls(const int ball_count)
{

	CUDAScene* scenery = scene_factory(ball_count, ball_count);

	int threads = 512;

	int blocks = ball_count / threads + 1;

	gen_random_balls << <blocks, threads >> > (scenery, ball_count);

	checkCudaErrors(cudaDeviceSynchronize());

	return scenery;
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


CUDAScene* single_ball()
{
	CUDAScene* scenery = scene_factory(1, 1);

	gen_single_ball << <1, 1>> > (scenery);

	checkCudaErrors(cudaDeviceSynchronize());

	return scenery;
}


__global__ void gen_single_ball(CUDAScene* const scene)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < 1)
	{
		vec3 center = vec3(3.f, 0.f, 0.f);
		float radius = 1.f;
		Material<CUDA_RNG>* mat = new Diffuse<CUDA_RNG>(vec3(1.f, 0.f, 0.f));

		(*scene->visibles)[id] = new CUDASphere(center, radius, mat);
		(*scene->materials)[id] = mat;
	}
}



CUDAScene* single_triangle()
{

	CUDAScene* scenery = scene_factory(1, 1);

	gen_single_triangle << <1, 1 >> > (scenery);

	checkCudaErrors(cudaDeviceSynchronize());

	return scenery;
}


__global__ void gen_single_triangle(CUDAScene* const scene)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id == 0)
	{
		vec3 a = vec3(0.f, 0.f, 1.f);

		vec3 b = vec3(0.f, 1.f, 1.f);

		vec3 c = vec3(1.f, 0.f, 1.f);

		vec3 points[3] = { a, b, c };

		Material<CUDA_RNG>* mat = new Metal<CUDA_RNG>(vec3(.5f, .2f, .2f), 0.1f);

		(*scene->visibles)[id] = new Triangle(points, mat);
		(*scene->materials)[id] = mat;

	}
}


Array<vec3>* cube_vertices(const vec3& translation = vec3(0.f, 0.f, 0.f))
{

	Array<vec3>* vertex_array = new Array<vec3>(8);

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			for (int k = 0; k < 2; k++)
			{
				(*vertex_array)[i + 2 * j + 4 * k] = vec3(i, j, k) + translation;
			}

	return vertex_array;
}

Array<uint32_t>* cube_indices()
{
	Array<uint32_t>* index_array = new Array<uint32_t>(36);

	int indices[36] = {
		0, 2, 1, 2, 3, 1,
		1, 5, 4, 0, 1, 4,
		4, 5, 7, 4, 7, 6,
		6, 7, 3, 3, 2, 6,
		1, 3, 5, 3, 7, 5,
		0, 4, 6, 0, 6, 2
	};

	for (int i = 0; i < 36; i++)
		(*index_array)[i] = indices[i];

	return index_array;
}

CUDAScene* single_cube()
{

	Array<vec3>* vertex_array = cube_vertices();

	Array<vec3>* const device_vertex_array = vertex_array->to_device();

	Array<uint32_t>* index_array = cube_indices();

	Array<uint32_t>* const device_index_array = index_array->to_device();

	Material<CUDA_RNG>* mat = new Diffuse<CUDA_RNG>(vec3(0.7f, 0.1f, 0.2f));

	Material<CUDA_RNG>* const device_mat = mat->to_device();

	CUDAScene* scenery = scene_factory(1,1);

	gen_single_cube << <1, 1 >> > (scenery, device_vertex_array, device_index_array, device_mat);

	checkCudaErrors(cudaDeviceSynchronize());

	delete vertex_array;
	delete index_array;
	delete mat;

	return scenery;
}


__global__ void gen_single_cube(CUDAScene* const scene, const Array<vec3>* const vertex_array, const Array<uint32_t>* const index_array, Material<CUDA_RNG>* const mat)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id == 0)
	{

		(*scene->visibles)[id] = new Mesh(vertex_array, index_array, mat);
		(*scene->materials)[id] = mat;
	}
}

template<typename T>
__host__ T* move_to_device(T* const obj)
{
	T* device_ptr = obj->to_device();

	delete obj;

	return device_ptr;
}


CUDAScene* n_cubes(const int& n)
{

	Array<CUDAVisible*>* visibles = new Array<CUDAVisible*>(n);
	Array<Array<vec3>*>* vertex_arrays = new Array<Array<vec3>*>(n);
	Array<Array<uint32_t>*>* index_arrays = new Array<Array<uint32_t>*>(n);
	Array<Material<CUDA_RNG>*>* material_array = new Array<Material<CUDA_RNG>*>(n);

	for (int i = 0; i < n; i++)
	{
		const Array<vec3>* vertex_array = cube_vertices(vec3(0.f, 0.f, 1.5f*i));

		(*vertex_arrays)[i] = vertex_array->to_device();

		delete vertex_array;

		Array<uint32_t>* index_array = cube_indices();

		(*index_arrays)[i] = index_array->to_device();

		delete index_array;

		(*material_array)[i] = Diffuse<CUDA_RNG>(vec3((float)i / (float)(n - 1), .5f, 1.f - (float)i / (float)(n - 1))).to_device();

	}

	Array<CUDAVisible*>* device_visibles = visibles->to_device();

	Array<Array<vec3>*>* device_vertex_arrays = vertex_arrays->to_device();

	Array<Array<uint32_t>*>* device_index_arrays = index_arrays->to_device();

	Array<Material<CUDA_RNG>*>* device_material_array = material_array->to_device();

	CUDAScene* scene = new CUDAScene();

	scene->visibles = device_visibles;

	scene->materials = device_material_array;

	scene->vertex_arrays = device_vertex_arrays;

	scene->index_arrays = device_index_arrays;

	gen_n_cubes << <1, n >> > (scene);

	checkCudaErrors(cudaDeviceSynchronize());

	delete visibles;

	delete vertex_arrays;

	delete index_arrays;

	delete material_array;

	return scene;

}

__global__ void gen_n_cubes(CUDAScene* const scene)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id < scene->visibles->size())
	{

		(*scene->visibles)[id] = new Mesh((*scene->vertex_arrays)[id], (*scene->index_arrays)[id], (*scene->materials)[id]);
	}

}



