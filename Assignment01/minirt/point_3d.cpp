#include "point_3d.h"

namespace minirt {

Point3D::Point3D() :
	x(0), y(0), z(0) {		
}

Point3D::Point3D(double x, double y, double z) :
	x(x), y(y), z(z) {		
}

Point3D::Point3D(double c) :
	x(c), y(c), z(c) {		
}	

Vector3D operator -(const Point3D &p1, const Point3D &p2) {
    return Vector3D {p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};
}

Point3D operator +(const Point3D &p, const Vector3D &v) {
    return Point3D {p.x + v.x, p.y + v.y, p.z + v.z};
}

Point3D operator -(const Point3D &p, const Vector3D &v) {
    return Point3D {p.x - v.x, p.y - v.y, p.z - v.z};
}

std::istream& operator >> (std::istream &in, Point3D &p) {
    return in >> p.x >> p.y >> p.z;
}

}
