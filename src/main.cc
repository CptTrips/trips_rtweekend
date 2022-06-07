#include <iostream>
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
#include "CUDAScenes.cuh"
//#include "Scenes.h"

const bool CUDA_ENABLED = false;

// Resolution
const int res_multiplier = 1;
const int w = res_multiplier*16;
const int h = res_multiplier*9;

// Samples per pixel
const int spp = 2;

// Ray bounce limit
const int max_bounce = 2;
int bounce_count = 0;


void test_visible_size(std::vector<std::unique_ptr<Visible>>& scene)
{

    for (auto& v : scene)
    {
        std::cout << "Size of " << typeid(*v).name() << " " << v->size() << std::endl;
    }

}

int main() {

	// Arrange scene
	// 

	//single_ball();

	//const int ball_count = 9;
	//random_balls(ball_count);

	//quadric_test();

	//grid_balls();


	//std::vector<std::unique_ptr<Visible>> scene = grid_balls();

    //const int ball_count = 16;
    //CUDAVisible** const random_balls_scene = random_balls(ball_count);

	CUDAVisible** const single_ball_scene = single_ball();

	// Place camera
	/*
	vec3 camera_origin = 2.*vec3(1., .5, -3.0);
	vec3 camera_target = vec3(0., -.3, -2.);
	vec3 camera_up = vec3(0.0, 1.0, 0.0);
	*/

	vec3 camera_origin = vec3(0.f, 0.f, 0.f);
	vec3 camera_target = vec3(1., 0., 0.);
	vec3 camera_up = vec3(0.0, 1.0, 0.0);

	// Right-handed
	float vfov = 3.;

	float aspect_ratio = float(w) / float(h);

	float focus_distance = (camera_target - camera_origin).length();

	float aperture = 0.1;

	Camera view_cam(camera_origin, camera_target, camera_up, vfov, aspect_ratio, focus_distance, aperture);


	// Draw scene

	//CPURayTracer cpu_ray_tracer(spp, max_bounce);
	GPURayTracer gpu_ray_tracer(spp, max_bounce);

	FrameBuffer* frame_buffer = gpu_ray_tracer.render(h, w, single_ball_scene, 1, view_cam);


	// Write scene
	bmp_write(frame_buffer->buffer, h, w, "output.bmp");

	cudaDeviceSynchronize();

	// Exit
	delete frame_buffer;

	cudaFree(single_ball_scene);

	return 0;

}
