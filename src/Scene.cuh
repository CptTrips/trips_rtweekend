#pragma once

#include <string>
#include <vector>

#include "materials\diffuse.cuh"
#include "CUDAScene.cuh"
#include "CUDASceneGenerators.cuh"
#include "visibles\Mesh.cuh"

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags

#include <device_launch_parameters.h>

class Scene
{

	CUDAScene* cuda_scene = NULL;

	std::vector<aiMesh*> ai_meshes;

	Diffuse<CUDA_RNG> default_material;

	Array<vec3>** vertex_library = NULL;
    const Array<vec3>** device_vertex_library = NULL;

	Array<uint32_t>** index_library = NULL;
	const Array<uint32_t>** device_index_library = NULL;

	uint32_t* device_material_library = NULL;

	void process_node(aiNode* node, const aiScene* scene);

	void send_meshes();

	void send_mesh_data(const aiMesh* const m, const uint32_t& i);

	void send_material();


public:
	Scene() : default_material(vec3(.5f, .5f, .5f)) {}
	Scene(const Scene& s) = delete;
	Scene& operator=(const Scene& s) = delete;
	~Scene();
	Scene(std::string scene_path);
	CUDAScene* to_device();
};

__device__ const Array<vec3>* get_mesh_vertices(const Array<vec3>** const vertex_library, const uint32_t& mesh_id);

__device__ const Array<uint32_t>* get_mesh_indices(const Array<uint32_t>** const index_library, const uint32_t& mesh_id);

__device__ const Material<CUDA_RNG>* get_mesh_material(const CUDAScene* const scene, const uint32_t& mesh_id, const uint32_t* const material_library);

__global__ void register_material(CUDAScene* const scene, const Material<CUDA_RNG>* const material);

__global__ void fill_scene(CUDAScene* const scene, const uint32_t* const material_library, const Array<vec3>** const vertex_library, const Array<uint32_t>** const index_library);
