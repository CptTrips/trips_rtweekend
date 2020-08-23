#include "lens.h"

Lens::Lens() {}

Lens::Lens(float R1, float R2, float d, vec3 x0, vec3 dir, Material* m){

    mat3 Q1 = mat3(1.f);
    vec3 P1 = x0 + (d/2.+R1)*dir;
    float R_1 = R1*R1;
    sphere_1 = Quadric(Q1, P1, R_1, m, true);

    mat3 Q2 = mat3(1.f);
    vec3 P2 = x0 - (d/2.+R2)*dir;
    float R_2 = R2*R2;
    sphere_2 = Quadric(Q2, P2, R_2, m, true);

    float dir_projector_elements[9];
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            dir_projector_elements[3*i + j] = dir[i] * dir[j];
        }
    }
    mat3 Q3 = mat3(1.f) - mat3(dir_projector_elements);;
    vec3 P3 = x0 + R1*dir;
    float R_3 = R1*R1;
    cylinder = Quadric(Q1, P1, R_1, m, true);
}

Lens::~Lens() {
    //delete material;
}

bool Lens::intersect(const Ray& r, float tmin, float tmax, Intersection& ixn) const {

    std::cout << "hi";

    Intersection ixn1, ixn2, ixn3;

    // Get intersections with sphere_1 (if none return false)
    if (!sphere_1.intersect(r, tmin, tmax, ixn1)) {
        return false;
    }

    // Get intersection with sphere_2 (if none return false)
    if (!sphere_2.intersect(r, tmin, tmax, ixn2)) {
        return false;
    }

    // Get intersection with cylinder
    if (!cylinder.intersect(r, tmin, tmax, ixn3)) {
        ixn3.t = MAXFLOAT;
        ixn3.p = vec3(MAXFLOAT, MAXFLOAT, MAXFLOAT);
    }

    // There are at most two intersections which are within the other two quadrics

    Intersection* valid_ixns[2];
    int counter = 0;

    if (enclosed_by(ixn1.p, sphere_2) && enclosed_by(ixn1.p, cylinder)) {

        valid_ixns[counter] = &ixn1;
        counter++;
    }

    if (enclosed_by(ixn2.p, sphere_1) && enclosed_by(ixn2.p, cylinder)) {
        valid_ixns[counter] = &ixn2;
        counter++;
    }

    if (enclosed_by(ixn3.p, sphere_1) && enclosed_by(ixn3.p, sphere_2)) {
        valid_ixns[counter] = &ixn3;
        counter++;
    }
    std::cout << "valid ixns: " << counter;

    switch (counter) {
        case 0:
            return false;
            //break;
        case 1:
            ixn = *valid_ixns[0];
            std::cout << "lens hit";
            return true;
            //break;
        case 2:
            if (valid_ixns[0]->t < valid_ixns[1]->t) {
                ixn = *valid_ixns[0];
            } else {
                ixn = *valid_ixns[1];
            }
            std::cout << "lens hit";
            return true;
    }

    // Should throw and exception here really
    return false;

}


bool enclosed_by(const vec3 x, const Quadric& q) {
    // Does point x lie "inside" q?

    float f = dot(x, q.Q*x) + dot(q.P,x) + q.R;

    return f < 0;
}
