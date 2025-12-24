#pragma once

#include "point_3d.h"
#include "vector_3d.h"

namespace minirt {

struct Ray {
    Point3D origin;
    Vector3D direction;

    Ray();
    Ray(const Point3D &origin, const Vector3D &direction);

    Point3D fromOrigin(double distance) const;
};

}
