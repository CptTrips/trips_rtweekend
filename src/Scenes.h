#pragma once
#include <memory>
#include <vector>
#include "visibles/visible.h"
#include "materials/material.h"
#include "visibles/sphere.cuh"
#include "materials/metal.h"
#include "materials/diffuse.h"
#include "materials/dielectric.h"

CPU_RNG rng = CPU_RNG();

std::vector<std::unique_ptr<Visible>> random_balls(const int ball_count) {

    std::vector<std::unique_ptr<Visible>> scenery;

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

    
    scenery.push_back(std::make_unique<Sphere>(center, r, m));
  }

  return scenery;
}

/*
Scene single_ball() {

  // Set up scene
  const int ball_count = 2;

  std::vector<Visible*> scenery;

  Diffuse* grass = new Diffuse(vec3(.42, .62, 0.05));
  Metal* ground = new Metal(vec3(0.7, 0.7, 0.7), 0.);
  Metal* ball = new Metal(vec3(.9, .1, .1), 0.1);

  scenery.push_back(new Sphere(vec3(0, 0, -1), .5, ball));

  float big_radius = 500.;
  scenery.push_back(new Sphere(vec3(0, -big_radius-.5, -1), big_radius, grass));

  return Scene(scenery);
}
*/

std::vector<std::unique_ptr<Visible>> grid_balls()
{
  // Set up scene
  const int ball_count = 10;

  std::vector<std::unique_ptr<Visible>> scenery;

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

      scenery.push_back(std::make_unique<Sphere>(center, r, mat));
    }
  }

  //Metal* ground = new Metal(vec3(0.7, 0.7, 0.7), 0.8);
  Diffuse* ground = new Diffuse(vec3(0.6, 0.6, 0.6));

  float big_radius = 500.;
  scenery.push_back(std::make_unique<Sphere>(vec3(0, -big_radius-r, -1), big_radius, ground));

  return scenery;
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


