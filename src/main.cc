#include <iostream>
#include <functional>
#include "rand.h"
#include "ray.h"
#include "..\include\palette.h"
#include "visible_list.h"
#include "sphere.h"
#include <quadric.h>
#include <lens.h>
#include "metal.h"
#include "diffuse.h"
#include "dielectric.h"
#include "float.h"


// Resolution
const int res_multiplier = 2;
const int w = res_multiplier*160;
const int h = res_multiplier*90;

// Samples per pixel
const int spp = 16;

// Ray bounce limit
const int max_bounce = 3;
int bounce_count = 0;

// static here means internal linkage, rng is only visible to this file
RNG rng = RNG();

struct Camera
{
  vec3 origin;
  vec3 z_dir;
  vec3 x_dir;
  vec3 y_dir;
};

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

vec3 shade_ray(const Ray& ray, VisibleList* const scene) {

  if (bounce_count == max_bounce) {
    return vec3(0., 0., 0.);
  }

  bounce_count += 1;

  Intersection ixn;

  if (scene->intersect(ray, 1e-12, MAXFLOAT, ixn)) {

    Material* active_material = ixn.material;

    Ray scatter_ray;

    active_material->bounce(ray, ixn, scatter_ray);

    return active_material->albedo * shade_ray(scatter_ray, scene);

  }

  vec3 sky_color = draw_sky(ray);

  return sky_color;
}

Ray cast_ray(float r, float c, float h, float w, Camera& cam) {

  float x = (c  - w / 2.0) / h; // x goes from -w/(2h) to w/(2h)
  float y = r / h - 0.5; // y goes from -.5 to .5

  vec3 ray_dir = x * cam.x_dir + y * cam.y_dir + cam.z_dir;

  return Ray(cam.origin, ray_dir);
}

vec3 exposure(const vec3 spectral_power_density) {
    // Determines the RGB colour of the pixel given an incident spectral power
    // density. (Perhaps make a member function of the camera.)

    vec3 col = vec3();

    float max_power = 1;

    for (int i=0; i<3; i++) {
        col[i] = spectral_power_density[i] / max_power;
        col[i] = std::min(col[i], 1.f);
    }

    return col;
}

void shade_buffer(char* buffer, const int h, const int w, VisibleList* const scene) {

  vec3 camera_origin = vec3(0., 0., -3.0);
  vec3 z_dir = normalise(-camera_origin);

  // lens test
  //vec3 camera_origin = vec3(0, 0, -6.9);
  //vec3 z_dir = normalise(vec3(0, 0, 1));
  vec3 x_dir = normalise(cross(z_dir, vec3(0.0, 1.0, 0.0)));
  vec3 y_dir = normalise(cross(x_dir, z_dir));
  float zoom = 0.75;

  Camera view_cam = {camera_origin, zoom*z_dir, x_dir, y_dir};

  for (int r=0; r<h; r++)
  {
    for (int c=0; c<w; c++)
    {

      vec3 spectral_power_density = vec3(0., 0., 0.);
      vec3 col = vec3(0.,0.,0.);

      for (int s=0; s<spp; s++) {


        float u = float(r) + rng.sample();
        float v = float(c) + rng.sample();

        bounce_count = 0;
        Ray primary = cast_ray(float(u), float(v), float(h), float(w), view_cam);

        spectral_power_density += shade_ray(primary, scene);
      }

      spectral_power_density /= float(spp);

      col = exposure(spectral_power_density);

      gamma_correction(col);

      buffer[r*w*3 + c*3 + 0] = int(255.99*col.r());
      buffer[r*w*3 + c*3 + 1] = int(255.99*col.g());
      buffer[r*w*3 + c*3 + 2] = int(255.99*col.b());
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

  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {

      vec3 dx = vec3(1., 0., 0.);
      vec3 dz = vec3(0., 0., -1.);
      vec3 o = vec3(0., 0., -2.) - dx - dz;

      vec3 center = o + i*dx + j*dz;


      vec3 color = vec3(0.15, 0.15, 0.15);
      color[j] = 0.9;

      Material* mat;

      switch (i) {
        case 0: { // Diffuse
          mat = new Diffuse(color);
          break;
                }
        case 1: { // Dielectric
          mat = new Dielectric(vec3(1.,1.,1.), 1.2);
          break;
                }
        case 2: { // Rough metal
          mat = new Metal(color, 0.1);
          break;
                }
      }

      scenery[3*i+j] = new Sphere(center, r, mat);
    }
  }

  Metal* ground = new Metal(vec3(0.7, 0.7, 0.7), 0.8);

  float big_radius = 500.;
  scenery[9] = new Sphere(vec3(0, -big_radius-r, -1), big_radius, ground);

  return new VisibleList(scenery, ball_count);
}

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

int main() {

  //single_ball();

  //const int ball_count = 9;
  //random_balls(ball_count);

  VisibleList* scene = lens();
  //quadric_test();
  //grid_balls();


  // Draw scene
  char* buffer = new char[h*w*3];

  shade_buffer(buffer, h, w, scene);

  delete scene;

  bmp_write(buffer, h, w, "output.bmp");

  delete [] buffer;

  return 0;

}
