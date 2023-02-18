#include "Camera.h"
#include "FrameBuffer.cuh"
//#include "CPURayTracer.h"
#include "GPURayTracer.cuh"
#include "CUDASceneGenerators.cuh"
#include "CUDAScene.cuh"
#include "SceneLoader.cuh"

#include "test_kernel.cuh"

#include "float.h"

#include <json.hpp>
#include <palette.h>

#include <unordered_map>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>


using json = nlohmann::json;

const bool CUDA_ENABLED = false;

SceneLoader host_scene;

enum SceneID {random_balls_id, single_ball_id, single_triangle_id, single_cube_id, n_cubes_id, backpack_id,
	triangle_carpet_id, rtweekend_id, json_id};

std::unordered_map<std::string, SceneID> scene_name_to_id = {
	{"random_balls", random_balls_id}
	,{"single_ball", single_ball_id}
	,{"single_triangle", single_triangle_id}
	,{"single_cube", single_cube_id}
	,{"n_cubes", n_cubes_id}
	,{"backpack", backpack_id}
	,{"triangle_carpet", triangle_carpet_id}
	,{"rtweekend", rtweekend_id}
	,{"json", json_id}
};


CUDAScene* load_scene(std::string scene_name, const json& j)
{
	CUDAScene* scene;
	
	auto it = scene_name_to_id.find(scene_name);

	if (it != scene_name_to_id.end())
	{
		SceneID scene_id = it->second;


		switch (scene_id)
		{
		case random_balls_id:
			scene = random_balls(j["random_balls"]["ball_count"]);
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
		case n_cubes_id:
			scene = n_cubes(j["n_cubes"]["n"]);
			break;
		case triangle_carpet_id:
			scene = triangle_carpet(j["triangle_carpet"]["n"]);
			break;
		case backpack_id:
			host_scene = SceneLoader(std::string(j["backpack_path"]));
			scene = host_scene.to_device();
			break;
		case rtweekend_id:
			scene = rtweekend(j["rtweekend"]["attempts"], j["rtweekend"]["seed"]);
			break;
		case json_id:
			scene = new CUDAScene(std::string(j["json"]["scene_path"]));
			break;
		}

	}
	else // try to load scene as a file 
	{
		host_scene = SceneLoader(scene_name);
		scene = host_scene.to_device();
	}

	return scene;
}


int main()
{

	// Get scene configuration
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

	const float min_free_path = j["min_free_path"];


    const int ball_count = j["random_balls"]["ball_count"];

	std::string scene_name = j["scene_name"];

	CUDAScene* const scene = load_scene(scene_name, j);

	json camera_json;

	if (j[scene_name].contains("camera"))
		camera_json = j[scene_name]["camera"];
	else
		camera_json = j["camera"];

	std::vector<float> origin_vec = camera_json["origin"];
	vec3 camera_origin = vec3(origin_vec[0], origin_vec[1], origin_vec[2]);

	std::vector<float> target_vec = camera_json["target"];
	vec3 camera_target = vec3(target_vec[0], target_vec[1], target_vec[2]);

	std::vector<float> up_vec = camera_json["up"];
	vec3 camera_up = vec3(up_vec[0], up_vec[1], up_vec[2]);


	float vfov = camera_json["vfov"];

	float aspect_ratio = float(w) / float(h);

	float focus_distance = (camera_target - camera_origin).length();

	float aperture = camera_json["aperture"];

	Camera view_cam(camera_origin, camera_target, camera_up, vfov, aspect_ratio, focus_distance, aperture);


	// Draw scene

	//CPURayTracer cpu_ray_tracer(spp, max_bounce);
	GPURayTracer gpu_ray_tracer;

	GPURenderProperties render_properties{ h, w, spp, max_bounce, min_free_path};

	FrameBuffer* frame_buffer = gpu_ray_tracer.render(render_properties, view_cam);


	// Write scene
	bmp_write(frame_buffer->buffer, h, w, "output.bmp");

	cudaDeviceSynchronize();

	// Exit
	delete frame_buffer;

	// Delete visibles and materials
	// cudaFree/Delete arrays
	// cudaFree scene
	delete scene;

	return 0;

}
