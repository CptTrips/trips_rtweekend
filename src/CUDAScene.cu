#include "CUDAScene.cuh"

__host__ CUDAScene::CUDAScene()
{
	visibles = NULL;

	materials = NULL;

	index_arrays = NULL;

	vertex_arrays = NULL;
}

__host__ CUDAScene::CUDAScene(UnifiedArray<CUDAVisible*>* const visibles, UnifiedArray<Material<CUDA_RNG>*>* const materials)
	//: visibles(visibles), materials(materials)
{
	this->visibles = visibles;
	this->materials = materials;

	index_arrays = NULL;

	vertex_arrays = NULL;

}

CUDAScene::CUDAScene(const unsigned int visible_count, const unsigned int material_count)
{

	visibles = new UnifiedArray<CUDAVisible*>(visible_count);

	materials = new UnifiedArray<Material<CUDA_RNG>*>(material_count);

}

__host__ CUDAScene::CUDAScene(const std::string& fp)
{
	std::ifstream input_file(fp);
	nlohmann::json j;
	input_file >> j;

	unsigned int visible_count = j.size();

	visibles = new UnifiedArray<CUDAVisible*>(visible_count);

	materials = new UnifiedArray<Material<CUDA_RNG>*>(visible_count);

	UnifiedArray<CUDASphere>* host_spheres = new UnifiedArray<CUDASphere>(visible_count);

	for (unsigned int i = 0; i < visible_count; i++)
	{
		auto json_visible = j[i];

		if (json_visible["type"] == "Sphere")
		{

			vec3 center = vec3(json_visible["center"][0], json_visible["center"][1], json_visible["center"][2]);

			(*host_spheres)[i] = CUDASphere(center, (float)json_visible["radius"], NULL);

			auto json_material = json_visible["material"];

			vec3 albedo = vec3(json_material["albedo"][0], json_material["albedo"][1], json_material["albedo"][2]);

			(*materials)[i] = Material<CUDA_RNG>(
				albedo,
				json_material["diffuse"],
				json_material["metal"],
				json_material["dielectric"],
				json_material["roughness"],
				json_material["refractive_index"]
			).to_device();
		}
	}


	checkCudaErrors(cudaDeviceSynchronize());

	int threads = 512;

	int blocks = visible_count / threads + 1;

	instantiate_spheres << <blocks, threads >> > (this, host_spheres);

	checkCudaErrors(cudaDeviceSynchronize());

	delete host_spheres;
}

__global__ void instantiate_spheres(CUDAScene* const scene, const UnifiedArray<CUDASphere>* const spheres)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id < spheres->size())
	{
		(*scene->visibles)[id] = new CUDASphere(
			(*spheres)[id].center,
			(*spheres)[id].radius,
			(*scene->materials)[id]
		);
	}
}

/*
__device__ CUDAScene::CUDAScene(const CUDAScene& cs)
{
	visibles = cs.visibles;
	materials = cs.materials;
	index_arrays = cs.index_arrays;
	vertex_arrays = cs.vertex_arrays;
}

__device__ CUDAScene& CUDAScene::operator=(const CUDAScene& cs)
{
	if (this == &cs)
		return *this;

	set_visibles(cs.visibles);

	set_materials(cs.materials);

	return *this;
}

__device__ CUDAScene::CUDAScene(CUDAScene&& cs)
{

	set_visibles(cs.visibles);

	cs.visibles = NULL;

	set_materials(cs.materials);

	cs.materials = NULL;

}

// Move assignment
__device__ CUDAScene& CUDAScene::operator=(CUDAScene&& cs) 
{

	if (this == &cs)
		return *this;

	set_visibles(cs.visibles);

	cs.visibles = NULL;

	set_materials(cs.materials);

	cs.materials = NULL;

	return *this;
}
*/

__host__ CUDAScene::~CUDAScene()
{

	delete_visibles();

	delete_materials();

	delete_vertex_arrays();

	delete_index_arrays();

}

__host__ void CUDAScene::set_visibles(UnifiedArray<CUDAVisible*>* const new_visibles)
{
	delete_visibles();

	visibles = new_visibles;
}

__host__ void CUDAScene::set_materials(UnifiedArray<Material<CUDA_RNG>*>* const new_materials)
{

	delete_materials();

	materials = new_materials;
}

__global__ void cuda_delete_visibles(UnifiedArray<CUDAVisible*>* visibles)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id < visibles->size())
		delete (*visibles)[id];

}

__host__ void CUDAScene::delete_visibles()
{
	int threads = 512;

	int blocks = visibles->size() / threads + 1;

	if (visibles)
		cuda_delete_visibles<<<blocks, threads>>>(visibles);


	checkCudaErrors(cudaDeviceSynchronize());

	delete visibles;

}

__host__ void CUDAScene::delete_materials()
{
	if (materials)
	{
		for (uint32_t i = 0; i < materials->size(); i++)
			cudaFree((*materials)[i]);

		cudaFree(materials);
	}
}


__host__ void CUDAScene::delete_vertex_arrays()
{
	if (vertex_arrays)
	{

		for (int i = 0; i < vertex_arrays->size(); i++)
			cudaFree((*vertex_arrays)[i]);

		cudaFree(vertex_arrays);
	}

}

__host__ void CUDAScene::delete_index_arrays()
{
	if (index_arrays)
	{
		for (int i = 0; i < index_arrays->size(); i++)
			cudaFree((*index_arrays)[i]);

		cudaFree(index_arrays);
	}

}

