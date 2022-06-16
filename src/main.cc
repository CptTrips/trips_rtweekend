#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>
#include "rand.h"
#include "ray.cuh"
#include "..\include\palette.h"
//#include "visibles/sphere.cuh"
#include "float.h"
#include "Camera.h"
#include "FrameBuffer.cuh"
//#include "CPURayTracer.h"
#include "GPURayTracer.cuh"
#include <typeinfo>
#include "CUDASceneGenerators.cuh"
#include "CUDAScene.cuh"
#include "..\include\json.hpp"

const bool CUDA_ENABLED = false;



void test_visible_size(std::vector<std::unique_ptr<Visible>>& scene)
{

    for (auto& v : scene)
    {
        std::cout << "Size of " << typeid(*v).name() << " " << v->size() << std::endl;
    }

}

int main() {

	// read a JSON file
	using json = nlohmann::json;
	std::ifstream i("config.json");
	json j;
	i >> j;

	// Resolution
	const int w = j["x_res"];
	const int h = j["y_res"];

	// Samples per pixel
	const int spp = j["spp"];

	// Ray bounce limit
	const int max_bounce = j["max_bounce"];

	// Arrange scene
	// 

    const int ball_count = j["random_balls"]["ball_count"];
	CUDAScene* const scene = random_balls(ball_count);

	//CUDAVisible** const scene = single_ball();
	//const int scene_size = 1;

	/*
	CUDAVisible** const scene = single_triangle();
	const int scene_size = 1;
	*/


	// Place camera
	// Grid Balls
	/*
	vec3 camera_origin = 2.*vec3(1., .5, -3.0);
	vec3 camera_target = vec3(0., -.3, -2.);
	vec3 camera_up = vec3(0.0, 1.0, 0.0);
	*/

	// Single Ball
	/*
	vec3 camera_origin = vec3(0.f, 0.f, 0.f);
	vec3 camera_target = vec3(1., 0., 0.);
	vec3 camera_up = vec3(0.0, 1.0, 0.0);
	*/


	// Random Balls
	std::vector<float> origin_vec = j["camera"]["origin"];
	vec3 camera_origin = vec3(origin_vec[0], origin_vec[1], origin_vec[2]);

	std::vector<float> target_vec = j["camera"]["target"];
	vec3 camera_target = vec3(target_vec[0], target_vec[1], target_vec[2]);

	std::vector<float> up_vec = j["camera"]["up"];
	vec3 camera_up = vec3(up_vec[0], up_vec[1], up_vec[2]);


	// Right-handed
	float vfov = j["camera"]["vfov"];

	float aspect_ratio = float(w) / float(h);

	float focus_distance = (camera_target - camera_origin).length();

	float aperture = j["camera"]["aperture"];

	Camera view_cam(camera_origin, camera_target, camera_up, vfov, aspect_ratio, focus_distance, aperture);


	// Draw scene

	//CPURayTracer cpu_ray_tracer(spp, max_bounce);
	GPURayTracer gpu_ray_tracer;

	GPURenderProperties render_properties{ h, w, spp, max_bounce };

	FrameBuffer* frame_buffer = gpu_ray_tracer.render(render_properties, scene, view_cam);


	// Write scene
	bmp_write(frame_buffer->buffer, h, w, "output.bmp");

	cudaDeviceSynchronize();

	// Exit
	delete frame_buffer;

	// Delete visibles and materials
	// cudaFree/Delete arrays
	// cudaFree scene
	teardown_scene(scene);

	return 0;

}
