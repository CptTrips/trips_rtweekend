#include <iostream>
#include <functional>
#include <algorithm>
#include "rand.h"
#include "ray.h"
#include "..\include\palette.h"
#include "visible_list.h"
#include "sphere.h"
#include "quadric.h"
#include "lens.h"
#include "metal.h"
#include "diffuse.h"
#include "dielectric.h"
#include "float.h"
#include "Camera.h"
#include "FrameBuffer.h"


const bool CUDA_ENABLED = false;

// Resolution
const int res_multiplier = 2;
const int w = res_multiplier*160;
const int h = res_multiplier*90;

// Samples per pixel
const int spp = 128;

// Ray bounce limit
const int max_bounce = 4;
int bounce_count = 0;

RNG rng = RNG();



void gamma_correction(vec3& col) {

  col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
}

vec3 draw_sky(const Ray& ray) {

  vec3 unit_dir = normalise(ray.direction());
  float t = 0.5*(unit_dir.y() + 1.0);
  return (1.0 - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

vec3 draw_night_sky(const Ray& ray) {
    static vec3 sun_dir = normalise(vec3(0.1, 0.075, -1));
    static vec3 sun_spd = 500*vec3(1.0, .98, .7);

    if (dot(ray.direction(), sun_dir) > 0.999) {
        return sun_spd;
    } else {
        return vec3(0, 0, 0.002);
    }
}

vec3 draw_red_star(const Ray& ray) {
    static vec3 star_dir = vec3(0,1,0);
    static vec3 star_spd = 100*vec3(1,0,0);

    if (dot(ray.direction(), star_dir) > 0.9) {
        return star_spd;
    } else {
        return vec3(0.5, 0.7, 1.0);
    }
}

vec3 shade_ray(const Ray& ray, const VisibleList& scene) {

  if (bounce_count == max_bounce) {
    return vec3(0., 0., 0.);
  }

  bounce_count += 1;

  Intersection ixn;

  if (scene.intersect(ray, 1e-12, FLT_MAX, ixn)) {

    Material* active_material = ixn.material;

    Ray scatter_ray;

    active_material->bounce(ray, ixn, scatter_ray);

    return active_material->albedo * shade_ray(scatter_ray, scene);

  }

  vec3 sky_color = draw_sky(ray);

  return sky_color;
}


vec3 exposure(const vec3 spectral_power_density, const float max_power = 1) {
    // Determines the RGB colour of the pixel given an incident spectral power
    // density. (Perhaps make a member function of the camera.)

    vec3 col = spectral_power_density;

    for (int i=0; i<3; i++) {
        col[i] /= max_power;
        col[i] = (std::min)(col[i], 1.f);
    }

    return col;
}

void shade_buffer(FrameBuffer& fb, const VisibleList& scene, const Camera& view_cam) {


  for (int r=0; r<fb.h; r++)
  {
    for (int c=0; c<fb.w; c++)
    {

      vec3 spectral_power_density = vec3(0., 0., 0.);
      vec3 col = vec3(0.,0.,0.);

      for (int s=0; s<spp; s++) {


        float u = (float(r) + rng.sample()) / float(fb.h);
        float v = (float(c) + rng.sample()) / float(fb.w);

        bounce_count = 0;
        Ray primary = view_cam.cast_ray(u, v);

        spectral_power_density += shade_ray(primary, scene);
      }

      spectral_power_density /= float(spp);

      col = exposure(spectral_power_density);

      gamma_correction(col);

      fb.set_pixel(r, c, col);
    }
  }
}

VisibleList* random_balls(const int ball_count) {

  Visible** scenery = new Visible*[ball_count];

  for (int i=0; i<ball_count; i++) {

    float r = 0.33; // ball radius

    float xrange = 2.;
    float yrange = 1.25;
    float zrange = 1.5;

    float zoffset = -1.;

    vec3 center = vec3(xrange*(2.*rng.sample()-1), yrange*(2.*rng.sample()-1), zoffset - zrange*rng.sample());

    vec3 color = vec3(rng.sample(), rng.sample(), rng.sample());

    float roughness = rng.sample();

    // Randomize the material
    Material* m;

    if (rng.sample() > 0.5) {

      m = new Metal(color, roughness);

    } else {

      m = new Diffuse(color);

    }

    scenery[i] = new Sphere(center, r, m);
  }

  return new VisibleList(scenery, ball_count);
}

VisibleList* single_ball() {

  // Set up scene
  const int ball_count = 2;

  Visible** scenery = new Visible*[ball_count];

  Diffuse* grass = new Diffuse(vec3(.42, .62, 0.05));
  Metal* ground = new Metal(vec3(0.7, 0.7, 0.7), 0.);
  Metal* ball = new Metal(vec3(.9, .1, .1), 0.1);

  scenery[0] = new Sphere(vec3(0, 0, -1), .5, ball);

  float big_radius = 500.;
  scenery[1] = new Sphere(vec3(0, -big_radius-.5, -1), big_radius, grass);

  return new VisibleList(scenery, ball_count);
}

VisibleList* grid_balls() {

  // Set up scene
  const int ball_count = 10;

  Visible** scenery = new Visible*[ball_count];

  float r = 0.3;

  vec3 dx = vec3(1., 0., 0.);
  vec3 dz = vec3(0., 0., -1.);
  vec3 o = vec3(0., 0., -2.) - dx - dz;

  vec3 white = vec3(1., 1., 1.);

  float color_mix = 0.35;
  float metal_color_mix = 0.75;
  float index = 1.52;

  Material* mat;

  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {


      vec3 center = o + i*dx + j*dz;

      vec3 color = 0.15 * white;
      color[j] = 0.9;

      switch (i) {
        case 0: { // Diffuse
          mat = new Diffuse(color);
          break;
                }
        case 1: { // Dielectric
          mat = new Dielectric((1. - color_mix)*vec3(1., 1., 1.) + color_mix*color, index);
          break;
                }
        case 2: { // Rough metal
          mat = new Metal((1. - metal_color_mix)*white + metal_color_mix*color, 0.1);
          break;
                }
      }

      scenery[3*i+j] = new Sphere(center, r, mat);
    }
  }

  //Metal* ground = new Metal(vec3(0.7, 0.7, 0.7), 0.8);
  Diffuse* ground = new Diffuse(vec3(0.6, 0.6, 0.6));

  float big_radius = 500.;
  scenery[9] = new Sphere(vec3(0, -big_radius-r, -1), big_radius, ground);

  return new VisibleList(scenery, ball_count);
}

/*
VisibleList* quadric_test() {

    vec3 offset;

    Visible** scenery = new Visible*[3];

    //ellipse
    offset = vec3(0,0,-5);
    mat3 Q_ellipse = mat3(2, 0, 0,
                          0, 2, 0.,
                          0, 0., 10);
    vec3 P_ellipse = -2 * Q_ellipse.T()*offset;
    float R_ellipse = -1. + dot(offset, (Q_ellipse * offset));

    Dielectric* mat_ellipse = new Dielectric(vec3(1,1,1), 1.5);

    scenery[0] = new Quadric(Q_ellipse, P_ellipse, R_ellipse, mat_ellipse);

    // cylinder
    offset = vec3(-1, -1, 0);
    mat3 Q_cylinder = mat3(1, 0., 0.,
                           0., 1, 0.,
                           0., 0., 0);
    vec3 P_cylinder = -2. * Q_cylinder.T()*offset;
    float R_cylinder = -.01 + dot(offset, (Q_cylinder * offset));
    Diffuse* mat_cylinder = new Diffuse(vec3(0,0,0.04));
    scenery[1] = new Quadric(Q_cylinder, P_cylinder, R_cylinder, mat_cylinder);

    // hyperboloid
    offset = vec3(0, 0, 0);
    mat3 Q_hyperboloid = mat3(4, 0, 0,
                              0, -1, 0,
                              0, 0, 4);
    vec3 P_hyperboloid = -2. * Q_hyperboloid.T() * offset;
    float R_hyperboloid = .0025 + dot(offset, (Q_hyperboloid * offset));
    Metal* mat_hyperboloid = new Metal(vec3(.9,.9,1), 0.05);
    scenery[2] = new Quadric(Q_hyperboloid, P_hyperboloid, R_hyperboloid, mat_hyperboloid);

    return new VisibleList(scenery, 3);

}

VisibleList* lens() {

    Visible** scenery = new Visible*[2];

    // Lens
    float n = 2; // Refractive index
    float R1 = 1;
    float R2 = 1;
    float d = 0.2;
    vec3 x0 = vec3(0., 0., 0.);
    vec3 dir = vec3(0., 0., 1.);

    Dielectric* mat_ellipse = new Dielectric(vec3(.9,.1,.1), n);

    scenery[0] = new Lens(R1, R2, d, x0, dir, mat_ellipse);

    // Paper plane
    mat3 Q_plane = mat3(0.f);
    vec3 P_plane = vec3(0, -2, 0);
    float R_plane = 0;

    Diffuse* mat_plane = new Diffuse(vec3(1,0,0));

    scenery[1] = new Quadric(Q_plane, P_plane, R_plane, mat_plane);

    return new VisibleList(scenery, 2);

}
*/

int main() {

  // Arrange scene
  // 
  //single_ball();

  //const int ball_count = 9;
  //random_balls(ball_count);

  VisibleList* scene = grid_balls();
  //quadric_test();
  //grid_balls();


  // Place camera
  vec3 camera_origin = 2.*vec3(1., .5, -3.0);
  vec3 camera_target = vec3(0., -.3, -2.);
  vec3 camera_up = vec3(0.0, 1.0, 0.0);

  // lens test
  //vec3 camera_origin = vec3(0, 0, -6.9);
  //vec3 z_dir = normalise(vec3(0, 0, 1));
  // Right-handed

  float vfov = 3.;

  float aspect_ratio = float(w) / float(h);

  float focus_distance = (camera_target - camera_origin).length();

  float aperture = 0.1;

  Camera view_cam(camera_origin, camera_target, camera_up, vfov, aspect_ratio, focus_distance, aperture);


  // Draw scene
  FrameBuffer* frame_buffer = new FrameBuffer(h, w);

  shade_buffer(*frame_buffer, *scene, view_cam);


  // Write scene
  bmp_write(frame_buffer->buffer, h, w, "output.bmp");


  // Exit
  delete scene;

  delete frame_buffer;

  return 0;

}
