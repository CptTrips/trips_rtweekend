#pragma once
#include <vector>
#include "visibles/visible.h"
#include "rendering/FrameBuffer.cuh"
#include "geometry/Camera.h"
#include "maths/vec3.cuh"
#include "maths/rand.cuh"
#include "materials/material.h"
#include "geometry/Scene.h"


class CPURayTracer
{

	const int spp;

	const int max_bounce;

	int bounce_count;

	const float tmin;

	const float tmax;

	vec3 shade_ray(const Ray& ray, const std::vector<std::unique_ptr<Visible>>& scene);

	vec3 exposure(const vec3 spectral_power_density, const float max_power = 1) const;

	void gamma_correction(vec3& col) const;

	vec3 draw_sky(const Ray& ray) const;

	CPU_RNG rng;

public:

	CPURayTracer(const int spp, const int max_bounce, const float tmin=1e-12, const float tmax=FLT_MAX) : spp(spp), max_bounce(max_bounce), bounce_count(0), tmin(tmin), tmax(tmax), rng(CPU_RNG()) {}

	FrameBuffer* render(const int h, const int w, const std::vector<std::unique_ptr<Visible>>& scene, const Camera& camera);

	std::unique_ptr<Intersection> nearest_intersection(const Ray& ray, const std::vector<std::unique_ptr<Visible>>& scene) const;
};