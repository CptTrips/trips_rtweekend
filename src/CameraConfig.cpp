#include "CameraConfig.h"

const std::string CameraConfig::CAMERA_KEY = "camera";
const std::string CameraConfig::ORIGIN_KEY = "origin";
const std::string CameraConfig::TARGET_KEY = "target";
const std::string CameraConfig::UP_KEY = "up";
const std::string CameraConfig::XRES_KEY = "xRes";
const std::string CameraConfig::YRES_KEY = "yRes";
const std::string CameraConfig::VERTICAL_FOV_KEY = "verticalFOV";
const std::string CameraConfig::APERTURE_KEY = "aperture";
const std::string CameraConfig::FOCUS_DISTANCE_KEY = "focusDistance";

CameraConfig::CameraConfig(const nlohmann::json& configJSON)
	: xRes(configJSON[XRES_KEY])
	, yRes(configJSON[YRES_KEY])
	, verticalFOV(configJSON[CAMERA_KEY][VERTICAL_FOV_KEY])
	, aperture(configJSON[CAMERA_KEY][APERTURE_KEY])
	, focusDistance(configJSON[CAMERA_KEY][FOCUS_DISTANCE_KEY])
{

	origin = vectorToVec(configJSON[CAMERA_KEY][ORIGIN_KEY]);

	target = vectorToVec(configJSON[CAMERA_KEY][TARGET_KEY]);

	up = vectorToVec(configJSON[CAMERA_KEY][UP_KEY]);
}

vec3 CameraConfig::vectorToVec(const std::vector<float>& vec)
{

	return vec3(vec[0], vec[1], vec[2]);
}
