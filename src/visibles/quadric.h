#ifndef QUADRIC_H
#define QUADRIC_H

#include "visible.h"
#include "material.h"
#include "mat3.h"


class Quadric : public Visible
{
    public:
        Quadric();
        Quadric(mat3 Q, vec3 P, float R, Material* m, bool offset=false);
        ~Quadric();
        virtual std::unique_ptr<Intersection> intersect(const Ray& r, float tmin, float tmax) const;
        vec3 normal(vec3 x) const;

        mat3 Q;
        vec3 P;
        float R;

        Material* material;
};

#endif // QUADRIC_H
