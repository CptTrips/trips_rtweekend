#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CUDAScanTest.cuh"

#include "geometry/Camera.cuh"
#include "rendering/FrameBuffer.cuh"
//#include "CPURayTracer.h"
#include "rendering/GPURayTracer.cuh"

#include "geometry/Scene.cuh"
#include "geometry/SceneBuilders.cuh"
//#include "SceneLoader.cuh"

//#include "test_kernel.cuh"

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

//SceneLoader host_scene;

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

int doctest_main(int argc, char** argv)
{
    doctest::Context context;

    // !!! THIS IS JUST AN EXAMPLE SHOWING HOW DEFAULTS/OVERRIDES ARE SET !!!

    // defaults
    //context.addFilter("test-case-exclude", "*math*"); // exclude test cases with "math" in their name
    //context.setOption("abort-after", 5);              // stop test execution after 5 failed assertions
    context.setOption("order-by", "file");            // sort the test cases by their name

    context.applyCommandLine(argc, argv);

    // overrides
    //context.setOption("no-breaks", true);             // don't break in the debugger when assertions fail

    int res = context.run(); // run

    if(context.shouldExit()) // important - query flags (and --exit) rely on the user doing this
        return res;          // propagate the result of the tests
    
    return res; // the result from doctest is propagated here as well
}

/*
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
*/

json loadConfig(std::string configPath)
{

	std::ifstream configStream(configPath);

	json configJSON;

	configStream >> configJSON;

	return configJSON;
}

void runRayTracer()
{

	json configJSON = loadConfig("config.json");

	/*
    const int ball_count = j["random_balls"]["ball_count"];

	std::string scene_name = j["scene_name"];

	CUDAScene* const scene = load_scene(scene_name, j);
	*/

	Camera camera(configJSON);

	//TestSceneBuilder sceneBuilder(20.f, 2.f);

	GridSceneBuilder sceneBuilder(3, 1.f);

	Scene scene{ sceneBuilder.buildScene() };

	// Draw scene

	//CPURayTracer cpu_ray_tracer(spp, max_bounce);
	GPURayTracer gpu_ray_tracer(configJSON);

	std::shared_ptr<FrameBuffer> frameBuffer = gpu_ray_tracer.render(scene, camera);

	// Write scene
	bmp_write(frameBuffer->buffer, frameBuffer->h, frameBuffer->w, "output.bmp");

	cudaDeviceSynchronize();
}

int main(int argc, char** argv)
{

	int doctestResult = doctest_main(argc, argv);

	runRayTracer();

	return doctestResult;
}
