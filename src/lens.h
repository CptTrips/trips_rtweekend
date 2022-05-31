#ifndef LENS_H
#define LENS_H

#include <cfloat>
#include "visible.h"
#include "material.h"
#include "quadric.h"

// Biconvex lens
class Lens : public Visible
{
    public:
        Lens();
        Lens(float R1, float R2, float d, vec3 x0, vec3 dir, Material* m);
        ~Lens();
        virtual std::unique_ptr<Intersection> intersect(const Ray& r, float tmin, float tmax, Intersection& ixn) const;
        vec3 normal(vec3 x) const;

        Quadric sphere_1;
        Quadric sphere_2;
        Quadric cylinder;

};

bool enclosed_by(const vec3 x, const Quadric& q);

#endif // LENS_H
