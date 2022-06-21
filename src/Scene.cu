#include "Scene.cuh"

Assimp::Importer Scene::ai_importer;

Scene::Scene(std::string scene_path) : default_material(Diffuse<CUDA_RNG>(vec3(0.5f, 0.5f, 0.5f)))
{

	ai_scene = ai_importer.ReadFile(scene_path, aiProcess_Triangulate);

	if (!ai_scene || ai_scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !ai_scene->mRootNode)
	{
		std::cout << "ERROR::ASSIMP::" << ai_importer.GetErrorString() << std::endl;
		throw;
	}

    process_node(ai_scene->mRootNode, ai_scene);


}

Scene& Scene::operator=(Scene&& s)
{
	cuda_scene = s.cuda_scene;
	s.cuda_scene = NULL;

	ai_meshes = s.ai_meshes;

	ai_scene = s.ai_scene;
    s.ai_scene = NULL;

    default_material = s.default_material;

    vertex_library = s.vertex_library;
    s.vertex_library = NULL;

    device_vertex_library = s.device_vertex_library;
    s.device_vertex_library = NULL;

    index_library = s.index_library;
    s.index_library = NULL;

    device_index_library = s.device_index_library;
    s.device_index_library = NULL;

    device_material_library = s.device_material_library;
	s.device_material_library = NULL;

    device_mat = s.device_mat;
    s.device_mat = NULL;

    return *this;
}

CUDAScene* Scene::to_device()
{


	cuda_scene = scene_factory(ai_meshes.size(), 1);

    // Send material

    send_material();

    send_meshes();

    int threads = 512;

    int blocks = ai_meshes.size() / threads + 1;

    fill_scene << <blocks, threads >> > (cuda_scene, device_material_library, device_vertex_library, device_index_library);

    checkCudaErrors(cudaPeekAtLastError());

	return cuda_scene;
}

void Scene::process_node(aiNode* node, const aiScene* scene)
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

void Scene::send_meshes()
{

    vertex_library = new Array<vec3>*[ai_meshes.size()];

    index_library = new Array<uint32_t>*[ai_meshes.size()];

    for (unsigned int i = 0; i < ai_meshes.size(); i++)
    {

        send_mesh_data(ai_meshes[i], i);

    }

    cudaMalloc(&device_vertex_library, sizeof(Array<vec3>*) * ai_meshes.size());

    cudaMalloc(&device_index_library, sizeof(Array<uint32_t>*) * ai_meshes.size());

    cudaMalloc(&device_material_library, sizeof(uint32_t) * ai_meshes.size());

    cudaMemcpy(device_vertex_library, vertex_library, sizeof(Array<vec3>*) * ai_meshes.size(), cudaMemcpyHostToDevice);

    cudaMemcpy(device_index_library, index_library, sizeof(Array<uint32_t>*) * ai_meshes.size(), cudaMemcpyHostToDevice);

    cudaMemset(device_material_library, 0, sizeof(uint32_t) * ai_meshes.size());

    cudaDeviceSynchronize();

}


Scene::~Scene()
{

    if (device_vertex_library) cudaFree(device_vertex_library);
    if (device_index_library) cudaFree(device_index_library);
    if (device_material_library) cudaFree(device_material_library);

    for (int i = 0; i < ai_meshes.size(); i++)
    {
        if (vertex_library)
        {
            checkCudaErrors(cudaFree(vertex_library[i]->get_data()));
            checkCudaErrors(cudaFree(vertex_library[i]));
        }
        if (index_library)
        {
            checkCudaErrors(cudaFree(index_library[i]->get_data()));
            checkCudaErrors(cudaFree(index_library[i]));
        }
    }

    if (vertex_library) delete vertex_library;
    if (index_library) delete index_library;

    if (device_mat)
        checkCudaErrors(cudaFree(device_mat));

    if (cuda_scene) teardown_scene(cuda_scene);

    if (ai_scene) delete ai_scene;
}

void Scene::send_mesh_data(const aiMesh* const m, const uint32_t& mesh_id)
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

    vertex_library[mesh_id] = device_vertex_array;

    delete vertex_array;


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
            (*index_array)[i + j] = ai_face.mIndices[j];

    }

    // Send indices

    Array<uint32_t>* device_index_array = index_array->to_device();

    // Register vertex array pointer with index library

    index_library[mesh_id] = device_index_array;

    delete index_array;


    // Material 

}


void Scene::send_material()
{

    cudaMallocManaged(&device_mat, sizeof(Diffuse<CUDA_RNG>));

    memcpy(device_mat, &default_material, sizeof(Diffuse<CUDA_RNG>));

    register_material << <1, 1 >> > (cuda_scene, device_mat);
}


__global__ void register_material(CUDAScene* const scene, const Material<CUDA_RNG>* const material)
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


__global__ void fill_scene(CUDAScene* const scene, const uint32_t* const material_library, const Array<vec3>** const vertex_library, const Array<uint32_t>** const index_library)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < scene->visibles->size())
    {

        const Array<vec3>* vertex_array = get_mesh_vertices(vertex_library, id);

        const Array<uint32_t>* index_array = get_mesh_indices(index_library, id);

        const Material<CUDA_RNG>* const material = get_mesh_material(scene, id, material_library);

        (*scene->visibles)[id] = new Mesh(vertex_array, index_array, material);

    }

}
