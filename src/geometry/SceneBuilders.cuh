#pragma once

#include "geometry/Scene.cuh"
/*
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "visibles/CUDAVisible.cuh"
#include "visibles/CUDASphere.cuh"
#include "visibles/Triangle.cuh"
#include "visibles/Mesh.cuh"
#include <curand_kernel.h>
#include "maths/rand.cuh"
#include "utility/Error.cuh"
#include "CUDAScene.cuh"
#include "memory/UnifiedArray.cuh"

#define my_cuda_seed 1234
*/



class SceneBuilder
{

	virtual void setMeshArrays(MeshFinder& mesh) = 0;

	virtual void setSphereArrays(Scene& scene) = 0;

	void allocateSphereArrays(Scene& scene);

protected:
	uint64_t vertexCount, triangleCount, sphereCount;

public:
	constexpr SceneBuilder(uint64_t vertexCount, uint64_t triangleCount, uint64_t sphereCount);

	Scene buildScene();

};

class TestSceneBuilder : public SceneBuilder
{

	virtual void setMeshArrays(MeshFinder& mesh) override;

	virtual void setSphereArrays(Scene& scene) override;

	float floorSize{ 1000.f };

	float bigRadius{ 1000.f };

	static constexpr uint64_t testSceneVertexCount{ 4 };
	static constexpr uint64_t testSceneTriangleCount{ 2 };
	static constexpr uint64_t testSceneSphereCount{ 2 };

public:
	TestSceneBuilder();
	TestSceneBuilder(float floorSize, float bigRadius);
};

class GridSceneBuilder : public SceneBuilder
{

	virtual void setMeshArrays(MeshFinder& mesh) override;

	virtual void setSphereArrays(Scene& scene) override;

	inline static uint64_t calculateVertexCount(uint32_t gridLength);
	inline static uint64_t calculateTriangleCount(uint32_t gridLength);
	inline static uint64_t calculateSphereCount(uint32_t gridLength);

	uint32_t gridLength;

	uint64_t gridLengthLong;

	float scale, radius;

public:
	GridSceneBuilder(uint32_t gridLength, float scale);
};

/*
CUDAScene* scene_factory(const int visible_count, const int material_count);

CUDAScene* rtweekend(int attempts = 22, int seed = 1);

CUDAScene* single_ball();

__global__ void gen_single_ball(CUDAScene* const scene);


__global__ void gen_rtweekend(CUDAScene* scene, UnifiedArray<vec3>* device_centers);

CUDAScene* random_balls(const int ball_count);

__global__ void gen_random_balls(CUDAScene* const scene, const int ball_count);


CUDAScene* single_triangle();

__global__ void gen_single_triangle(CUDAScene* const scene);

Array<vec3>* cube_vertices(const vec3& translation);

Array<uint32_t>* cube_indices();


CUDAScene* single_cube();

__global__ void gen_single_cube(CUDAScene* const scene, const Array<vec3>* const vertex_array, const Array<uint32_t>* const index_array, Material<CUDA_RNG>* const mat);

CUDAScene* n_cubes(const int& n);

__global__ void gen_n_cubes(CUDAScene* const scene);

CUDAScene* triangle_carpet(const unsigned int& n);

__global__ void gen_carpet(CUDAScene* const scene);

void teardown_scene(CUDAScene* scene);

*/
