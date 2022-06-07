#pragma once
#include "device_launch_parameters.h"
#include "visibles/sphere.cuh"
#include "FrameBuffer.cuh"
#include <vector>
#include "Camera.h"
#include "curand_kernel.h"
#include "ray.cuh"
#include "Error.cuh"
#include "rand.h"


class GPURayTracer
{

	const int spp;

	const int max_bounce;

	int ray_count;

	int scene_size;

	CUDAVisible** gpu_scene;

	FrameBuffer* h_fb;

	Camera* d_cam;

	Ray* rays;

	vec3* ray_colours;

	void send_scene(const std::vector<std::unique_ptr<Visible>>& scene);

	void alloc_framebuffer(const int h, const int w);

	void generate_rays();

	void shade_rays();

	void render_rays();

public:

	GPURayTracer(const int spp, const int max_bounce) : spp(spp), max_bounce(max_bounce) {}

	FrameBuffer* render(const int h, const int w, CUDAVisible** const scene, const int scene_size, const Camera& camera);

};

__global__ void colour_space(FrameBuffer* const fb);

__global__ void cuda_render_rays(vec3* ray_colours, const int ray_count, FrameBuffer* const fb, const int spp);

__global__ void cuda_gen_rays(Ray* rays, const int ray_count, const Camera* const cam, const FrameBuffer* const fb, curandState* cr_state, const int spp);

__device__ Intersection* nearest_intersection(const Ray& ray, CUDAVisible** const scene, const int scene_size, const float tmin, const float tmax);

__device__ vec3 draw_sky(const Ray& ray);

__global__ void cuda_shade_ray(Ray* const rays, vec3* const ray_colours, const int ray_count, CUDAVisible** const scene, const int scene_size, const int max_bounce);
