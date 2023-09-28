#pragma once

#include <string>
#include <json.hpp>

#include <array>

#include <cstdint>

#include "maths/vec3.cuh"

struct CameraConfig
{

    static const std::string CAMERA_KEY;
    static const std::string ORIGIN_KEY;
    static const std::string TARGET_KEY;
    static const std::string UP_KEY;
    static const std::string XRES_KEY;
    static const std::string YRES_KEY;
    static const std::string VERTICAL_FOV_KEY;
    static const std::string APERTURE_KEY;
    static const std::string FOCUS_DISTANCE_KEY;

    CameraConfig(const nlohmann::json& configJSON);


    vec3 origin, target, up;

    uint16_t xRes, yRes;

    float verticalFOV, aperture, focusDistance;

private:

    static vec3 vectorToVec(const std::vector<float>& vec);
};
