#include "geometry/Camera.cuh"


Camera::Camera(
	const vec3& origin,
	const vec3& target,
	const vec3& up,
	uint16_t xRes,
	uint16_t yRes,
	float verticalFOV,
	float aperture,
	float focusDistance
)	: origin(origin)
    , verticalFOV(verticalFOV)
    , aspectRatio(static_cast<float>(xRes) / static_cast<float>(yRes))
    , focusDistance(focusDistance)
    , aperture(aperture)
{

	vec3 cameraZ = normalise(target - origin);
	vec3 cameraY = normalise(cross(cameraZ, up));
	vec3 cameraX = normalise(cross(cameraY, cameraZ));

	orientation = mat3(
	  cameraX[0], cameraX[1], cameraX[2],
	  cameraY[0], cameraY[1], cameraY[2],
	  cameraZ[0], cameraZ[1], cameraZ[2]
	);
}


Camera::Camera(CameraConfig config)
	: Camera(
		config.origin
		, config.target
		, config.up
		, config.xRes
		, config.yRes
		, config.verticalFOV
		, config.aperture
		, config.focusDistance
	)
{

}



