#pragma once

#include "vector_3d.h"

#include <istream>

namespace minirt {

struct Point3D {
    double x;
    double y;
    double z;    

    Point3D();
    Point3D(double x, double y, double z);
    explicit Point3D(double c);
};	

Vector3D operator -(const Point3D &p1, const Point3D &p2);
Point3D operator +(const Point3D &p, const Vector3D &v);
Point3D operator -(const Point3D &p, const Vector3D &v);

std::istream& operator >> (std::istream &in, Point3D &p);

}
