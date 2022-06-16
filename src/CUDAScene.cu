#include "CUDAScene.cuh"

__host__ __device__ CUDAScene::CUDAScene()
{
	visibles = NULL;

	materials = NULL;
}

__device__ CUDAScene::CUDAScene(Array<CUDAVisible*>* const visibles, Array<Material<CUDA_RNG>*>* const materials)
	//: visibles(visibles), materials(materials)
{
	this->visibles = visibles;
	this->materials = materials;

}

__device__ CUDAScene::CUDAScene(const CUDAScene& cs)
{
	visibles = cs.visibles;
	materials = cs.materials;
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

__host__ __device__ CUDAScene::~CUDAScene()
{

	delete_visibles();

	delete_materials();
}


__device__ CUDAVisible* CUDAScene::operator[](const uint32_t i)
{
	return (*visibles)[i];
}

__device__ const CUDAVisible* CUDAScene::operator[](const uint32_t i) const
{
	return (*visibles)[i];
}

__device__ void CUDAScene::set_visibles(Array<CUDAVisible*>* const new_visibles)
{
	delete_visibles();

	visibles = new_visibles;
}

__device__ void CUDAScene::set_materials(Array<Material<CUDA_RNG>*>* const new_materials)
{

	delete_materials();

	materials = new_materials;
}
__device__ void CUDAScene::delete_visibles()
{
	if (visibles)
	{
		for (uint32_t i = 0; i < visibles->size(); i++)
			delete (*visibles)[i];

		delete visibles;
	}

}
__device__ void CUDAScene::delete_materials()
{
	if (materials)
	{
		for (uint32_t i = 0; i < materials->size(); i++)
			delete (*materials)[i];

		delete materials;
	}
}

__device__ uint32_t CUDAScene::size() const
{
	return (*visibles).size();
}
