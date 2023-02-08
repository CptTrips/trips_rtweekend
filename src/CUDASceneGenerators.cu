#include "CUDASceneGenerators.cuh"


CUDAScene* scene_factory(const int visible_count, const int material_count)
{

	CUDAScene* scene = new CUDAScene();

	UnifiedArray<CUDAVisible*>* visibles = new UnifiedArray<CUDAVisible*>(visible_count);

	UnifiedArray<Material<CUDA_RNG>*>* materials = new UnifiedArray<Material<CUDA_RNG>*>(material_count);

	scene->visibles = visibles;

	scene->materials = materials;

	return scene;
}


CUDAScene* rtweekend(int attempts, int seed)
{
	CPU_RNG rng = CPU_RNG(seed);

	std::vector<Material<CUDA_RNG>*> materials;

	std::vector<vec3> centers;

	for (int a = -attempts/2; a < attempts/2; a++)
	{
		for (int b = -attempts; b < attempts; b++)
		{

			float material_coin = rng.sample();

			vec3 center(a + 0.9f * rng.sample(), 0.2, b + 0.9 * rng.sample());

			if ((center - vec3(4, 0.2, 0)).length() > 0.9)
			{
				Material<CUDA_RNG>* mat;

				if (material_coin < 0.8)
				{
					vec3 albedo(rng.sample() * rng.sample(), rng.sample() * rng.sample(), rng.sample() * rng.sample());

					mat = new Diffuse<CUDA_RNG>(albedo);
				}
				else if (material_coin < 0.95)
				{
					vec3 albedo(0.5 * (1 + rng.sample()), 0.5 * (1 + rng.sample()), 0.5 * (1 + rng.sample()));

					float roughness = rng.sample();

					mat = new Metal<CUDA_RNG>(albedo, roughness);
				}
				else
				{
					vec3 albedo(1, 1, 1);

					mat = new Dielectric<CUDA_RNG>(albedo, 1.5);
				}

				materials.push_back(mat);
				centers.push_back(center);
			}
		}
	}

	unsigned int random_sphere_count = materials.size();

	CUDAScene* scene = new CUDAScene(random_sphere_count + 4, random_sphere_count + 4);

	UnifiedArray<vec3>* device_centers = new UnifiedArray<vec3>(random_sphere_count);

	for (unsigned int i = 0; i < random_sphere_count; i++)
	{
		(*scene->materials)[i] = materials[i]->to_device();

		checkCudaErrors(cudaDeviceSynchronize());

		delete materials[i];

		(*device_centers)[i] = centers[i];
	}

	materials.clear();

	Material<CUDA_RNG>* ground_mat = new Diffuse<CUDA_RNG>(vec3(0.5, 0.5, 0.5));
	
	Material<CUDA_RNG>* dielectric_mat = new Dielectric<CUDA_RNG>(vec3(1., 1., 1.), 1.5);

	Material<CUDA_RNG>* diffuse_mat = new Diffuse<CUDA_RNG>(vec3(0.4, 0.2, 0.1));

	Material<CUDA_RNG>* metal_mat = new Metal<CUDA_RNG>(vec3(0.7, 0.6, 0.5), 0.);

	(*scene->materials)[random_sphere_count] = ground_mat->to_device();
	(*scene->materials)[random_sphere_count + 1] = dielectric_mat->to_device();
	(*scene->materials)[random_sphere_count + 2] = diffuse_mat->to_device();
	(*scene->materials)[random_sphere_count + 3] = metal_mat->to_device();

	delete ground_mat;
	delete dielectric_mat;
	delete diffuse_mat;
	delete metal_mat;


	int threads = 1;

	int blocks = 1;

	gen_rtweekend << <blocks, threads >> > (scene, device_centers);

	checkCudaErrors(cudaDeviceSynchronize());

	return scene;
}


__global__ void gen_rtweekend(CUDAScene* scene, UnifiedArray<vec3>* device_centers)
{
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id == 0)
	{
		unsigned int visibles_count = scene->visibles->size();
		for (unsigned int i = 0; i < visibles_count - 4; i++)
		{
			(*scene->visibles)[i] = new CUDASphere((*device_centers)[i], 0.2f, (*scene->materials)[i]);
		}

		// ground
		(*scene->visibles)[visibles_count - 4] = new CUDASphere(vec3(0, -1000, 0), 1000, (*scene->materials)[visibles_count - 4]);

		// dielectric
		(*scene->visibles)[visibles_count - 3] = new CUDASphere(vec3(0, 1, 0), 1.0, (*scene->materials)[visibles_count - 3]);

		// diffuse
		(*scene->visibles)[visibles_count - 2] = new CUDASphere(vec3(-4, 1, 0), 1.0, (*scene->materials)[visibles_count - 2]);

		// metal
		(*scene->visibles)[visibles_count - 1] = new CUDASphere(vec3(4, 1, 0), 1.0, (*scene->materials)[visibles_count - 1]);
	}
}

CUDAScene* random_balls(const int ball_count)
{

	CUDAScene* scenery = new CUDAScene(ball_count, 2);

	for (unsigned int i = 0; i < ball_count; i++)
		(*scenery->materials)[i] = Material<CUDA_RNG>().to_device();

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
		Material<CUDA_RNG>* m = (*scene->materials)[id];

		if (rng.sample() > .5f) {

			*m = Metal<CUDA_RNG>(color, roughness);

		} else {

			*m = Diffuse<CUDA_RNG>(color);

		}

		(*scene->visibles)[id] = new CUDASphere(center, r, m);

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

	UnifiedArray<CUDAVisible*>* visibles = new UnifiedArray<CUDAVisible*>(n);
	UnifiedArray<Array<vec3>*>* vertex_arrays = new UnifiedArray<Array<vec3>*>(n);
	UnifiedArray<Array<uint32_t>*>* index_arrays = new UnifiedArray<Array<uint32_t>*>(n);
	UnifiedArray<Material<CUDA_RNG>*>* material_array = new UnifiedArray<Material<CUDA_RNG>*>(n);

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

	CUDAScene* scene = new CUDAScene();

	scene->visibles = visibles;

	scene->materials = material_array;

	scene->vertex_arrays = vertex_arrays;

	scene->index_arrays = index_arrays;

	gen_n_cubes << <1, n >> > (scene);

	checkCudaErrors(cudaDeviceSynchronize());

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


CUDAScene* triangle_carpet(const unsigned int& n)
{
	CUDAScene* scene = new CUDAScene();

	UnifiedArray<CUDAVisible*>* visibles = new UnifiedArray<CUDAVisible*>(1);
	UnifiedArray<Array<vec3>*>* vertex_arrays = new UnifiedArray<Array<vec3>*>(1);
	UnifiedArray<Array<uint32_t>*>* index_arrays = new UnifiedArray<Array<uint32_t>*>(1);
	UnifiedArray<Material<CUDA_RNG>*>* material_array = new UnifiedArray<Material<CUDA_RNG>*>(1);

	Array<vec3>* vertex_array = new Array<vec3>(n * n);

	Array<uint32_t>* index_array = new Array<uint32_t>(3 * 2 * (n - 1) * (n - 1));

	for (unsigned int i = 0; i < n; i++)
	{
		for (unsigned int j = 0; j < n; j++)
		{
			unsigned int vertex_index = n * i + j;

			(*vertex_array)[vertex_index] = vec3(0.f, 1.f * (float)j / (float)n, 1.f * (float)i / (float)n);

			if ((i < n - 1) && (j < n - 1))
			{
				unsigned int index_index = 6 * (n - 1) * i + 6 * j;
				(*index_array)[index_index] = vertex_index;
				(*index_array)[index_index + 1] = vertex_index + 1;
				(*index_array)[index_index + 2] = vertex_index + n;

				(*index_array)[index_index + 3] = vertex_index + n;
				(*index_array)[index_index + 4] = vertex_index + 1;
				(*index_array)[index_index + 5] = vertex_index + n + 1;

			}
		}
	}


	(*vertex_arrays)[0] = vertex_array->to_device();

	(*index_arrays)[0] = index_array->to_device();

	(*material_array)[0] = Diffuse<CUDA_RNG>(vec3(.5f, .5f, .5f)).to_device();

	checkCudaErrors(cudaDeviceSynchronize());

	scene->visibles = visibles;

	scene->materials = material_array;

	scene->vertex_arrays = vertex_arrays;

	scene->index_arrays = index_arrays;

	gen_carpet << <1, 1 >> > (scene);

	checkCudaErrors(cudaDeviceSynchronize());

	delete vertex_array;

	delete index_array;

	return scene;
}


__global__ void gen_carpet(CUDAScene* const scene)
{
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id == 0)
	{
		(*scene->visibles)[id] = new Mesh((*scene->vertex_arrays)[0], (*scene->index_arrays)[0], (*scene->materials)[0]);
	}
}



