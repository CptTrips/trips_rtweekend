#ifndef ROD_H
#define ROD_H

#include "visible.h"
#include "material.h"

class Rod : public Visible
{
	public:
		Rod();
		Rod(vec3 c, float r, vec3 l, Material* m);
		~Rod();
		std::unique_ptr<Intersection> intersect(const Ray& r, float tmin, float tmax) const;
		vec3 center;
		float radius;
		vec3 half_length_;
		Material* material;
};

#endif
