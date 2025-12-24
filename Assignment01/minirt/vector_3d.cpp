#include "vector_3d.h"

#include <cmath>

namespace minirt {

Vector3D::Vector3D() :
	x(0), y(0), z(0) {		
}

Vector3D::Vector3D(double x, double y, double z) :
	x(x), y(y), z(z) {		
}

Vector3D::Vector3D(double c) :
	x(c), y(c), z(c) {		
}

double Vector3D::lengthSquared() const {
    return x * x + y * y + z * z;
}

double Vector3D::length() const {
    return std::sqrt(lengthSquared());
}

Vector3D Vector3D::normalized() const {
    const auto len1 = 1.0 / length();
    return Vector3D {x * len1, y * len1, z * len1};
}

double Vector3D::dot(const Vector3D &v) const {
        return x * v.x + y * v.y + z * v.z;
}

Vector3D Vector3D::cross(const Vector3D &v) const {
    return Vector3D {y * v.z - z * v.y,
                     z * v.x - x * v.z,
                     x * v.y - y * v.x};
}

Vector3D operator -(const Vector3D &v1, const Vector3D &v2) {
    return Vector3D {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
}

Vector3D operator -(const Vector3D &v) {
    return Vector3D {-v.x, -v.y, -v.z};
}

Vector3D operator +(const Vector3D &v1, const Vector3D &v2) {
    return Vector3D {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
}

Vector3D operator *(const Vector3D &v, double c) {
    return Vector3D {c * v.x, c * v.y, c * v.z};
}

Vector3D operator *(double c, const Vector3D &v) {
    return v * c;
}

Vector3D operator /(const Vector3D &v, double c) {
    return v * (1.0 / c);
}

std::istream& operator >> (std::istream &in, Vector3D &v) {
    return in >> v.x >> v.y >> v.z;
}

}
