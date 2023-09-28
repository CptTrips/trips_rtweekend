#pragma once

#include "memory/UnifiedArray.cuh"

#include "maths/vec3.cuh"

#include "visibles/Triangle.cuh"

/// <summary>
/// Points to geometry and scattering data for a triangle mesh.
/// </summary>
struct MeshFinder
{

	UnifiedArray<vec3>* p_vertexArray;
	UnifiedArray<uint32_t>* p_indexArray;

	UnifiedArray<vec3>* p_faceNormalArray;

	UnifiedArray<vec3>* p_triangleColourArray;

};


/// <summary>
/// Owns geometry and scattering data for a triangle mesh.
/// </summary>
class MeshOwner
{

public:

	MeshOwner(
		uint64_t vertexCount,
		uint64_t triangleCount
	);

	void calculateFaceNormals();

	std::shared_ptr<MeshFinder> getFinder() const;

	std::shared_ptr<UnifiedArray<vec3>> p_vertexArray;
	std::shared_ptr<UnifiedArray<uint32_t>> p_indexArray;

	std::shared_ptr<UnifiedArray<vec3>> p_faceNormalArray;

	std::shared_ptr<UnifiedArray<vec3>> p_triangleColourArray;
};

