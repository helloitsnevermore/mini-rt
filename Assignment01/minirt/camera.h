#pragma once

#include "point_3d.h"
#include "vector_3d.h"
#include "ray.h"

namespace minirt {

class Camera {
public:
    Camera();
    Camera(const Point3D &viewPoint);
    Camera(const Point3D &viewPoint, const Point3D &target, const Vector3D &up = Vector3D {0, 1, 0});

    // Set camera's position only.
    void setViewPoint(const Point3D &point);

    // Set camera's position and orientation: camera is located at viewPoint and looks at target.
    void set(const Point3D &viewPoint, const Point3D &target, const Vector3D &up = Vector3D {0, 1, 0});

    // Shoot ray from the camera's view point, using provided scales for the camera's axes.
    Ray rayFrom(double dx, double dy, double dz) const;

    // Rotate camera aroung target on a given amount in degrees.
    void rotateAroundTarget(double degrees);
    Camera rotatedAroundTarget(double degrees) const;

    const Point3D& getViewPoint() const;
    const Point3D& getTarget() const;

    const Vector3D& getUnitX() const;
    const Vector3D& getUnitY() const;
    const Vector3D& getUnitZ() const;

private:
    // Camera's position.
    Point3D viewPoint {0, 0, -20};
    // Look a target.
    Point3D target {0, 0, 0};
    // Unit basis vectors for camera (left-handed system).
    Vector3D unitX {1, 0, 0};
    Vector3D unitY {0, 1, 0};
    Vector3D unitZ {0, 0, 1};
};

}
