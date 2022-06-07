#include "Camera.h"



Camera::Camera(
    const vec3& origin
    , const vec3& target
    , const vec3& up
    , float vfov
    , float aspect_ratio
    , float focus_distance
    , float aperture
) :
    origin(origin)
    , vfov(vfov)
    , aspect_ratio(aspect_ratio)
    , focus_distance(focus_distance)
    , aperture(aperture)
{
    
	vec3 z_dir = normalise(target - origin);
	vec3 y_dir = normalise(cross(z_dir, up));
	vec3 x_dir = normalise(cross(y_dir, z_dir));

	orientation = mat3(
	  x_dir[0], x_dir[1], x_dir[2],
	  y_dir[0], y_dir[1], y_dir[2],
	  z_dir[0], z_dir[1], z_dir[2]
	);
}


Ray Camera::cast_ray(const float& u, const float& v, CPU_RNG* rng) const
{ 
    // u and v run from 0 to 1

    // Right-handed so x is up :(
	float x = (u - 0.5) * vfov; // x goes from -vfov/2 to vfov/2
	float y = (v - 0.5) * vfov * aspect_ratio; // y goes from -w/(2h) to w/(2h)

    vec3 focus_offset = aperture * (2 * vec3(rng->sample(), rng->sample(), 0) - vec3(1., 1., 0.));

    vec3 cam_space_ray_dir = vec3(x, y, focus_distance) - focus_offset;

    vec3 ray_dir = orientation.T() * cam_space_ray_dir;

    return Ray(origin + orientation.T()*focus_offset, ray_dir);
}
