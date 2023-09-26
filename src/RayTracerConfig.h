#pragma once

#include <string>
#include "json.hpp"

struct RayTracerConfig
{

	static const std::string RAY_TRACER_KEY;
	static const std::string XRES_KEY;
	static const std::string YRES_KEY;
	static const std::string SAMPLES_PER_PIXEL_KEY;
	static const std::string MAX_BOUNCE_KEY;
	static const std::string MIN_FREE_PATH_KEY;

	unsigned int xRes, yRes;
	unsigned int spp;
	unsigned int maxBounce;
	float minFreePath;

	RayTracerConfig(nlohmann::json& config);
};

