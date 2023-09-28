#pragma once
#include "device_launch_parameters.h"
#include "maths/vec3.cuh"
#include "maths/mat3.h"
#include "geometry/ray.cuh"
#include "maths/rand.cuh"

#include <cstdint>

#include "utility/CameraConfig.h"

struct ImagePoint
{

    // u and v run from 0 to 1
	float u, v;
};

class Camera
{

    vec3 origin;
    mat3 orientation;
    float verticalFOV;
    float aspectRatio;
    float focusDistance; // Not to be confused with focal length. The plane at this distance from the origin will be in focus.
    float aperture;

public:

    Camera(CameraConfig config);

    Camera(
        const vec3& origin
        , const vec3& target
        , const vec3& up
        , uint16_t xRes
        , uint16_t yRes
        , float verticalFOV
        , float aperture
        , float focusDistance
    );

	template<class RNG>
    __device__ Ray castRay(const ImagePoint& p, RNG& rng) const
	{ 
		// Right-handed so x is up :(
		float x = (p.u - 0.5) * verticalFOV; // x goes from -verticalFOV/2 to verticalFOV/2
		float y = (p.v - 0.5) * verticalFOV * aspectRatio; // y goes from -w/(2h) to w/(2h)

		vec3 focalBlurOffset = aperture * (2 * vec3(rng.sample(), rng.sample(), 0) - vec3(1., 1., 0.));

		vec3 cam_space_ray_dir = vec3(x, y, focusDistance) - focalBlurOffset;

		vec3 rayDir = orientation.T() * cam_space_ray_dir;

		vec3 rayOrigin = origin + orientation.T() * focalBlurOffset;

		return Ray(rayOrigin, rayDir);
	}
};
