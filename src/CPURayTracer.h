#pragma once
#include <vector>
#include "visible.h"
#include "FrameBuffer.h"
#include "Camera.h"
#include "vec3.h"
#include "rand.h"
#include "material.h"
#include "Scene.h"


class CPURayTracer
{

	const int spp;

	const int max_bounce;

	int bounce_count;

	RNG rng;

	vec3 shade_ray(const Ray& ray, const std::vector<std::unique_ptr<Visible>>& scene);

	vec3 exposure(const vec3 spectral_power_density, const float max_power = 1) const;

	void gamma_correction(vec3& col) const;

	vec3 draw_sky(const Ray& ray) const;

public:

	CPURayTracer(const int spp, const int max_bounce) : spp(spp), max_bounce(max_bounce), rng(RNG()), bounce_count(0) {}

	void render(FrameBuffer& fb, const std::vector<std::unique_ptr<Visible>>& scene, const Camera& camera);

	std::unique_ptr<Intersection> nearest_intersection(const Ray& ray, const std::vector<std::unique_ptr<Visible>>& scene, const float tmin, const float tmax) const;
};