#include "SceneLoader.cuh"

Assimp::Importer SceneLoader::ai_importer;

SceneLoader::SceneLoader(std::string scene_path) : default_material(Diffuse<CUDA_RNG>(vec3(0.5f, 0.5f, 0.5f)))
{

	ai_scene = ai_importer.ReadFile(scene_path, aiProcess_Triangulate);

	if (!ai_scene || ai_scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !ai_scene->mRootNode)
	{
		std::cout << "ERROR::ASSIMP::" << ai_importer.GetErrorString() << std::endl;
		throw;
	}

    process_node(ai_scene->mRootNode, ai_scene);


}

SceneLoader& SceneLoader::operator=(SceneLoader&& s)
{
	cuda_scene = s.cuda_scene;
	s.cuda_scene = NULL;

	ai_meshes = s.ai_meshes;

	ai_scene = s.ai_scene;
    s.ai_scene = NULL;

    default_material = s.default_material;

    vertex_library = s.vertex_library;
    s.vertex_library = NULL;

    index_library = s.index_library;
    s.index_library = NULL;


    return *this;
}

CUDAScene* SceneLoader::to_device()
{

    cuda_scene = new CUDAScene();

    cuda_scene->materials = send_material();

    send_meshes();

    cuda_scene->visibles = new UnifiedArray<CUDAVisible*>(ai_meshes.size());

    // UnifiedArray with pointers to vertex_arrays, index_arrays and materials, one for each visible

    int threads = 512;

    int blocks = ai_meshes.size() / threads + 1;

    fill_scene << <blocks, threads >> > (cuda_scene);

    cudaDeviceSynchronize();

    checkCudaErrors(cudaPeekAtLastError());

	return cuda_scene;
}

void SceneLoader::process_node(aiNode* node, const aiScene* scene)
{
    // process each mesh located at the current node
    for (unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        // the node object only contains indices to index the actual objects in the scene. 
        // the scene contains all the data, node is just to keep stuff organized (like relations between nodes).

        ai_meshes.push_back(scene->mMeshes[node->mMeshes[i]]);
    }
    
    // after we've processed all of the meshes (if any) we then recursively process each of the children nodes
    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        process_node(node->mChildren[i], scene);
    }

}

void SceneLoader::send_meshes()
{

    vertex_library = new UnifiedArray<Array<vec3>*>(ai_meshes.size());

    index_library = new UnifiedArray<Array<uint32_t>*>(ai_meshes.size());

    for (unsigned int i = 0; i < ai_meshes.size(); i++)
    {

        send_mesh_data(ai_meshes[i], i);

    }

    cuda_scene->vertex_arrays = vertex_library;

    cuda_scene->index_arrays = index_library;

    cudaDeviceSynchronize();

}


SceneLoader::~SceneLoader()
{

    //if (ai_scene) delete ai_scene;
}

void SceneLoader::send_mesh_data(const aiMesh* const m, const uint32_t& mesh_id)
{

    unsigned int vertex_count = m->mNumVertices;

    Array<vec3>* vertex_array = new Array<vec3>(vertex_count);

    for (unsigned int i = 0; i < vertex_count; i++)
    {

        (*vertex_array)[i] = vec3(m->mVertices[i].x, m->mVertices[i].z, m->mVertices[i].z);

    }

    // Send vertices

    Array<vec3>* device_vertex_array = vertex_array->to_device();

    // Register vertex array pointer with vertex library

    (*vertex_library)[mesh_id] = device_vertex_array;

    delete vertex_array;

    cudaDeviceSynchronize();


    unsigned int index_count = m->mNumFaces * 3;

    Array<uint32_t>* index_array = new Array<uint32_t>(index_count);

    for (unsigned int i = 0; i < m->mNumFaces; i++)
    {

        aiFace ai_face = m->mFaces[i];

        if (ai_face.mNumIndices != 3)
        {
            std::cout << "ERROR: Face with more than 3 indices" << std::endl;
            delete index_array;
            throw;
        }

        for (int j = 0; j < 3; j++)
            (*index_array)[3*i + j] = ai_face.mIndices[j];

    }

    // Send indices

    Array<uint32_t>* device_index_array = index_array->to_device();

    // Register vertex array pointer with index library

    (*index_library)[mesh_id] = device_index_array;

    delete index_array;

    cudaDeviceSynchronize();
}


UnifiedArray<Material<CUDA_RNG>*>* SceneLoader::send_material()
{

    UnifiedArray<Material<CUDA_RNG>*>* material_array = new UnifiedArray<Material<CUDA_RNG>*>(1);

    (*material_array)[0] = default_material.to_device();

    return material_array;

}


__global__ void register_material(CUDAScene* const scene, Material<CUDA_RNG>* const material)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id == 0)
        (*scene->materials)[id] = material;

}


__device__ const Array<vec3>* get_mesh_vertices(const Array<vec3>** const vertex_library, const uint32_t& mesh_id)
{

    return vertex_library[mesh_id];

}

__device__ const Array<uint32_t>* get_mesh_indices(const Array<uint32_t>** const index_library, const uint32_t& mesh_id)
{

    return index_library[mesh_id];

}

__device__ const Material<CUDA_RNG>* get_mesh_material(const CUDAScene* const scene, const uint32_t& mesh_id, const uint32_t* const material_library)
{

    const uint32_t mat_id = material_library[mesh_id];

    return (*scene->materials)[mat_id];

}


__global__ void fill_scene(CUDAScene* const scene)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < scene->visibles->size())
    {

        (*scene->visibles)[id] = new Mesh((*scene->vertex_arrays)[id], (*scene->index_arrays)[id], (*scene->materials)[0]);

    }

}
