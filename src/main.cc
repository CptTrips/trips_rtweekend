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
#include <unordered_map>

const bool CUDA_ENABLED = false;



void test_visible_size(std::vector<std::unique_ptr<Visible>>& scene)
{

    for (auto& v : scene)
    {
        std::cout << "Size of " << typeid(*v).name() << " " << v->size() << std::endl;
    }

}

enum SceneID {random_balls_id, single_ball_id, single_triangle_id, single_cube_id };

std::unordered_map<std::string, SceneID> scene_name_to_id = {
	{"random_balls", random_balls_id}
	,{"single_ball", single_ball_id}
	,{"single_triangle", single_triangle_id}
	,{"single_cube", single_cube_id}
};

CUDAScene* load_scene(std::string scene_name, const int ball_count)
{
	
	auto it = scene_name_to_id.find(scene_name);

	if (it == scene_name_to_id.end())
		throw std::runtime_error("Invalid scene name");

	SceneID scene_id = it->second;

	CUDAScene* scene;

	switch (scene_id)
	{
	case random_balls_id:
		scene = random_balls(ball_count);
		break;
	case single_ball_id:
		scene = single_ball();
		break;
	case single_triangle_id:
		scene = single_triangle();
		break;
	case single_cube_id:
		scene = single_cube();
		break;
	}

	return scene;
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

	std::string scene_name = j["scene_name"];

	CUDAScene* const scene = load_scene(scene_name, ball_count);

	// CUDAScene* const scene = random_balls(ball_count);
	// CUDAScene* const scene = single_triangle();


	// Place camera
	// Grid Balls
	/*
	vec3 camera_origin = 2.*vec3(1., .5, -3.0);
	vec3 camera_target = vec3(0., -.3, -2.);
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
