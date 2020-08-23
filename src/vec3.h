#ifndef VEC3_H
#define VEC3_H

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3
{
  public:
    vec3() {}
    vec3(float e0, float e1, float e2){ e[0] = e0; e[1] = e1; e[2] = e2; }

    // Inline means the compiler will copy the function definition wherever
    // it's invoked, increasing program memory usage but saving execution time
    // by not moving the instruction pointer so much.
    //
    // const member functions guarantees that the function does not change the
    // object state, so is safe to invoke on a const Object o.

    inline float x() const {return e[0];}
    inline float y() const {return e[1];}
    inline float z() const {return e[2];}
    inline float r() const {return e[0];}
    inline float g() const {return e[1];}
    inline float b() const {return e[2];}

    // vec3& is a reference to a vec3, much like passing a dereference to a
    // const pointer
    inline const vec3& operator+() const { return *this; }
    inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    inline float operator[](int i) const { return e[i]; }
    inline float& operator[](int i) { return e[i]; }

    inline vec3& operator+=(const vec3 &v2);
    inline vec3& operator-=(const vec3 &v2);
    inline vec3& operator*=(const vec3 &v2);
    inline vec3& operator/=(const vec3 &v2);
    inline vec3& operator*=(const float t);
    inline vec3& operator/=(const float t);

    inline float length() const {
      return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    inline float squared_length() const {
      return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    inline void normalise();

    float e[3];
};

inline std::istream& operator>>(std::istream &is, vec3 &t) {
  is >> t.e[0] >> t.e[1] >> t.e[2];
  return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
  os << t.e[0] << " " << t.e[1] << " " << t.e[2];
  return os;
}

inline void vec3::normalise() {
  float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
  e[0] *= k; e[1] *= k; e[2] *= k;
}

inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

inline vec3 operator*(float t, const vec3 &v1) {
  return vec3(t*v1.e[0], t*v1.e[1], t*v1.e[2]);
}

inline vec3 operator/(const vec3 &v1, float t) {
  return vec3(v1.e[0]/t, v1.e[1]/t, v1.e[2]/t);
}

inline vec3 operator*(const vec3 &v1, float t) {
  return vec3(t*v1.e[0], t*v1.e[1], t*v1.e[2]);
}

inline float dot(const vec3 &v1, const vec3 &v2) {
  return v1.e[0]*v2.e[0] + v1.e[1]*v2.e[1] + v1.e[2]*v2.e[2];
}

inline vec3 cross(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1],
              v1.e[2]*v2.e[0] - v1.e[0]*v2.e[2],
              v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]);
}

inline vec3& vec3::operator+=(const vec3 &v) {
  e[0] += v.e[0];
  e[1] += v.e[1];
  e[2] += v.e[2];
  return *this;
}

inline vec3& vec3::operator*=(const vec3 &v) {
  e[0] *= v.e[0];
  e[1] *= v.e[1];
  e[2] *= v.e[2];
  return *this;
}

inline vec3& vec3::operator/=(const vec3 &v) {
  e[0] /= v.e[0];
  e[1] /= v.e[1];
  e[2] /= v.e[2];
  return *this;
}

inline vec3& vec3::operator-=(const vec3 &v) {
  e[0] -= v.e[0];
  e[1] -= v.e[1];
  e[2] -= v.e[2];
  return *this;
}

inline vec3& vec3::operator*=(const float t) {
  e[0] *= t;
  e[1] *= t;
  e[2] *= t;
  return *this;
}

inline vec3& vec3::operator/=(const float t) {
  float k = 1.0/t; // Not sure why we do it like this?

  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
  return *this;
}

inline vec3 normalise(vec3 v) {
  return v / v.length();
}

#endif
