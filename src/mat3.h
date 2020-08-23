#ifndef MAT3_H
#define MAT3_H

#include <math.h>
#include <stdlib.h>
#include <iostream>

class mat3
{
    public:
        mat3() {}
        mat3(float a00, float a01, float a02,
             float a10, float a11, float a12,
             float a20, float a21, float a22) {
             a[0] = a00; a[1] = a01; a[2] = a02;
             a[3] = a10; a[4] = a11; a[5] = a12;
             a[6] = a20; a[7] = a21; a[8] = a22;}
         mat3(float* a_in) { for (int i=0; i<9; i++) {a[i] = a_in[i];} }
         mat3(float a_diag) {
            for (int i=0; i<9; i++) {a[i] = 0.;}
            a[0] = a_diag; a[4] = a_diag; a[8] = a_diag; }

        inline float el(int i, int j) const { return a[3*i + j]; }
        inline void set_el(int i, int j, float a_ij) { a[3*i + j] = a_ij;}

        inline const mat3& operator+() const { return *this; }
        inline const mat3 operator-() const {
            float a_minus[9];
            for (int i=0; i<9; i++) { a_minus[i] = -a[i]; }
            return a_minus;
        }

        inline float operator[](int i) const { return a[i]; }
        inline float& operator[](int i) { return a[i]; }
        /* Need to define a mat3_row class to make this work
        inline float& operator[](int i, int j) { return el(i,j);}
        inline float operator[](int i, int j) { return el(i,j);}
        */

        inline mat3& operator+=(const mat3 &m2);
        inline mat3& operator-=(const mat3 &m2);
        inline mat3& operator*=(const mat3 &m2);

        inline mat3& operator+=(const float f);
        inline mat3& operator-=(const float f);
        inline mat3& operator*=(const float f);
        inline mat3& operator/=(const float f);

        inline mat3 T() const {
            mat3 a_T = mat3(0.);
            for (int i=0; i<3; i++) {
                for (int j=0; j<3; j++) {
                    a_T.set_el(j, i, this->el(i,j));
                }
            }
            return a_T;
        }

        float trace() const {
            float tr = 0.;

            for (int i=0; i<3; i++) {
                tr += this->el(i, i);
            }

            return tr;
        };

        float a[9];
};

inline std::istream& operator>>(std::istream &is, mat3 &m) {
  is >> m.a[0] >> m.a[1] >> m.a[2]
     >> m.a[3] >> m.a[4] >> m.a[5]
     >> m.a[6] >> m.a[7] >> m.a[8];
  return is;
}

inline std::ostream& operator<<(std::ostream &os, const mat3 &m) {
  os << "[" << m.a[0] << " " << m.a[1] << " " << m.a[2] << "\n"
            << m.a[3] << " " << m.a[4] << " " << m.a[5] << "\n"
            << m.a[6] << " " << m.a[7] << " " << m.a[8] << "]\n";
  return os;
}

inline mat3 operator+(const mat3 &m1, const mat3 &m2) {
    mat3 m_out = mat3(0.);
    for (int i=0; i<9; i++) {
        m_out[i] = m1[i] + m2[i];
    }
    return m_out;
}

inline mat3 operator-(const mat3 &m1, const mat3 &m2) {
  return m1 + (-m2);
}

inline mat3 operator*(const mat3 &m1, const mat3 &m2) {
    mat3 m_out = mat3(0.);

    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            float x = 0.;
            for (int k=0; k<3; k++) { x += m1.el(i, k) * m2.el(k, j); }

            m_out.set_el(i, j, x);
        }
    }

    return m_out;
}

inline vec3 operator*(const mat3 &m, const vec3 &v) {
    vec3 v_out = vec3(0., 0., 0.);
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            v_out.e[i] += m.el(i, j) * v.e[j];
        }
    }
    return v_out;
}

inline mat3 operator*(float t, const mat3 &m1) {
    return m1 * mat3(t);
}

inline mat3 operator/(const mat3 &m1, float t) {
    return m1 * mat3(1./t);
}

inline mat3 operator*(const mat3 &m1, float t) {
    return mat3(t) * m1;
}

#endif
