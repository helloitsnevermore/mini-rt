#pragma once

#include <istream>

namespace minirt {

struct Vector3D {
    double x;
    double y;
    double z;

    Vector3D();
    Vector3D(double x, double y, double z);
    explicit Vector3D(double c);

    double lengthSquared() const;
    double length() const;

    Vector3D normalized() const;

    double dot(const Vector3D &v) const;

    Vector3D cross(const Vector3D &v) const;
};

Vector3D operator -(const Vector3D &v1, const Vector3D &v2);
Vector3D operator -(const Vector3D &v);
Vector3D operator +(const Vector3D &v1, const Vector3D &v2);
Vector3D operator *(const Vector3D &v, double c);
Vector3D operator *(double c, const Vector3D &v);
Vector3D operator /(const Vector3D &v, double c);

std::istream& operator >> (std::istream &in, Vector3D &v);


}
