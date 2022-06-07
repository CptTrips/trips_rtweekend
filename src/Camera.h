#pragma once
#include "device_launch_parameters.h"
#include "vec3.cuh"
#include "mat3.h"
#include "ray.cuh"
#include "rand.h"

class Camera
{

public:

    vec3 origin;
    mat3 orientation;
    float vfov;
    float aspect_ratio;
    float focus_distance;
    float aperture;

    Camera(
        const vec3& origin
        , const vec3& target
        , const vec3& up
        , float vfov
        , float aspect_ratio
        , float focus_distance
        , float aperture
    );

    Ray cast_ray(const float& u, const float& v, CPU_RNG* rng) const;
};
