#pragma once

#include "point_3d.h"
#include "vector_3d.h"
#include "material.h"

namespace minirt {

class Ray;

class Sphere {
public:
    Sphere(const Point3D &position, double radius, const Material &material);
    Sphere(const Point3D &position, double radius = 1.0, const Color &color = Color(1));

    // Check intersection with the sphere.
    bool intersect(const Ray &ray, double &distance) const;

    // Outward normal to the sphere in the surface point.
    Vector3D normalTo(const Point3D &surfacePoint) const;

public:
    Point3D position;
    double radius;
    Material material;
};

}
