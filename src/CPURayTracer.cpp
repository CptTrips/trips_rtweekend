#include "CPURayTracer.h"

FrameBuffer* CPURayTracer::render(const int h, const int w, const std::vector<std::unique_ptr<Visible>>& scene, const Camera& camera)
{
    FrameBuffer* fb = new FrameBuffer(h, w);
  for (int r=0; r<h; r++)
  {
    for (int c=0; c<w; c++)
    {

      vec3 spectral_power_density = vec3(0., 0., 0.);
      vec3 col = vec3(0.,0.,0.);

      for (int s=0; s<spp; s++) {


        float u = (float(r) + rng.sample()) / float(h);
        float v = (float(c) + rng.sample()) / float(w);

        bounce_count = 0;
        Ray primary = camera.cast_ray(u, v);

        spectral_power_density += shade_ray(primary, scene);
      }

      spectral_power_density /= float(spp);

      col = exposure(spectral_power_density);

      gamma_correction(col);

      fb->set_pixel(r, c, col);
    }
  }

  return fb;
}

vec3 CPURayTracer::shade_ray(const Ray& ray, const std::vector<std::unique_ptr<Visible>>& scene) {

  if (bounce_count == max_bounce) {
    return vec3(0., 0., 0.);
  }

  bounce_count += 1;

  std::unique_ptr<Intersection> ixn_ptr = nearest_intersection(ray, scene);

  if (ixn_ptr) {

    const Material* active_material = ixn_ptr->material;

    Ray scatter_ray;

    active_material->bounce(ray, *ixn_ptr, scatter_ray);

    return active_material->albedo * shade_ray(scatter_ray, scene);

  }

  vec3 sky_color = draw_sky(ray);

  return sky_color;
}


void CPURayTracer::gamma_correction(vec3& col) const{

  col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
}

vec3 CPURayTracer::exposure(const vec3 spectral_power_density, const float max_power) const{
    // Determines the RGB colour of the pixel given an incident spectral power
    // density. (Perhaps make a member function of the camera.)

    vec3 col = spectral_power_density;

    for (int i=0; i<3; i++) {
        col[i] /= max_power;
        col[i] = (std::min)(col[i], 1.f);
    }

    return col;
}

std::unique_ptr<Intersection> CPURayTracer::nearest_intersection(const Ray& ray, const std::vector<std::unique_ptr<Visible>>& scene)const
{

    std::unique_ptr<Intersection> ixn;

    std::unique_ptr<Intersection> temp_ixn;

	bool any_intersect = false;

	float current_closest = tmax;

	for (const auto& it : scene)
    {

		temp_ixn = it->intersect(ray, tmin, current_closest);

		if (temp_ixn) {

			current_closest = temp_ixn->t;

            std::swap(ixn, temp_ixn);

		}
	}

	return ixn;

}

vec3 CPURayTracer::draw_sky(const Ray& ray) const{

  vec3 unit_dir = normalise(ray.direction());
  float t = 0.5f*(unit_dir.y() + 1.0f);
  return (1.0 - t)*vec3(1.f, 1.f, 1.f) + t*vec3(0.5f, 0.7f, 1.0f);
}
