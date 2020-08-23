#include "quadric.h"

Quadric::Quadric() {}

Quadric::Quadric(mat3 Q_in, vec3 P_in, float R_in, Material* m, bool offset) :
    Q(Q_in), material(m) {
        if (offset) {
            P = -2*Q_in*P_in;
            R = R_in + dot(P_in, Q_in * P_in);
        } else {
            P = P_in;
            R = R_in;
        }
}

Quadric::~Quadric() {
    delete material;
}

bool Quadric::intersect(const Ray& r, float tmin, float tmax, Intersection& ixn) const {

    vec3 d = r.direction();
    vec3 x0 = r.origin();
    vec3 Qd = Q * d;

    float a = dot(d, Qd);
    float b = 2 * dot(x0, Qd) + dot(P, d);
    float c = dot(P,x0) + dot(x0, Q*x0) + R;

    // Are there real solutions?
    float disc = b*b - 4 * a * c;

    if (disc > 0) {

        // Are the solutions within tmin and tmax?

        float sqrt_disc = sqrt(disc);

        float t = (-b - sqrt_disc) / (2 * a);

        if (t < tmax && t > tmin) {
          ixn.t = t;
          ixn.p = r.point_at(t);
          ixn.normal = this->normal(ixn.p);
          ixn.material = material;
          return true;
        }

        // If the material is transparent check the far solution
        if (!material->is_opaque()) {

          t += sqrt_disc/a;

          if (t < tmax && t > tmin) {
            ixn.t = t;
            ixn.p = r.point_at(t);
            ixn.normal = this->normal(ixn.p);
            ixn.material = material;
            return true;
          }
        }
    }
    return false;
}

vec3 Quadric::normal(vec3 x) const {

    // Assume x solves the quadric

    vec3 normal = (Q + Q.T()) * x + P;

    return normalise(normal);
}
