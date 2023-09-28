#include "utility/RayTracerConfig.h"

const std::string RayTracerConfig::XRES_KEY = "xRes";
const std::string RayTracerConfig::YRES_KEY = "yRes";
const std::string RayTracerConfig::RAY_TRACER_KEY = "rayTracer";
const std::string RayTracerConfig::SAMPLES_PER_PIXEL_KEY = "spp";
const std::string RayTracerConfig::MAX_BOUNCE_KEY = "maxBounce";
const std::string RayTracerConfig::MIN_FREE_PATH_KEY = "minFreePath";

RayTracerConfig::RayTracerConfig(nlohmann::json& config)
	: xRes(config[XRES_KEY])
	, yRes(config[YRES_KEY])
	, spp(config[RAY_TRACER_KEY][SAMPLES_PER_PIXEL_KEY])
	, maxBounce(config[RAY_TRACER_KEY][MAX_BOUNCE_KEY])
	, minFreePath(config[RAY_TRACER_KEY][MIN_FREE_PATH_KEY])
{
}
