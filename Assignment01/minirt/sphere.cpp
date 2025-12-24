#include "sphere.h"
#include "ray.h"

#include <cmath>

namespace minirt {

Sphere::Sphere(const Point3D &position, double radius, const Material &material) :
    position(position), radius(radius), material(material) {
}

Sphere::Sphere(const Point3D &position, double radius, const Color &color) :
    position(position), radius(radius), material(color) {
}

// Check intersection with the sphere.
bool Sphere::intersect(const Ray &ray, double &distance) const {
    Vector3D v = ray.origin - position;
    double d = v.dot(ray.direction);
    double discriminant = d * d - (v.dot(v) - radius * radius);
    if (discriminant < 0) {
        return false;
    }

    double t1 = -d + std::sqrt(discriminant);
    double t2 = -d - std::sqrt(discriminant);

    double t = 0;
    double epsilon = 1e-6;
    if (t1 < epsilon) {
        if (t2 < epsilon) {
            return false;
        } else {
            t = t2;
        }
    } else if (t2 < epsilon) {
        t = t1;
    } else {
        t = (t1 > t2) ? t2 : t1;
    }
    distance = t;
    return true;
}

Vector3D Sphere::normalTo(const Point3D &surfacePoint) const {
    return (surfacePoint - position).normalized();
}

}
