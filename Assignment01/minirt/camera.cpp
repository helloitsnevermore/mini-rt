#include "camera.h"

#include <cmath>

namespace minirt {

Camera::Camera() {}

Camera::Camera(const Point3D &viewPoint) {
    setViewPoint(viewPoint);
}

Camera::Camera(const Point3D &viewPoint, const Point3D &target, const Vector3D &up) {
    set(viewPoint, target, up);
}

void Camera::setViewPoint(const Point3D &point) {
    const auto shift = point - viewPoint;
    viewPoint = point;
    target = target + shift; // change target also
}

// Set camera's position and orientation: camera is located at viewPoint and looks at target.
void Camera::set(const Point3D &viewPoint, const Point3D &target, const Vector3D &up) {
    this->viewPoint = viewPoint;
    this->target = target;
    // Set up left-handed coordinate system for camera.
    unitZ = (target - viewPoint).normalized();
    unitX = up.cross(unitZ).normalized();
    unitY = unitZ.cross(unitX);
}

Ray Camera::rayFrom(double dx, double dy, double dz) const {
    Vector3D rayDirection = unitX * dx + unitY * dy + unitZ * dz;
    return Ray {viewPoint, rayDirection.normalized()};
}

void Camera::rotateAroundTarget(double degrees) {
    const auto pi = 3.14159265359;
    const auto dir = viewPoint - target;
    const auto radius = std::sqrt(dir.x * dir.x + dir.z * dir.z);
    const auto angle = std::atan2(dir.z, dir.x);
    const auto newAngle = angle + pi * degrees / 180.0;
    Point3D newViewPoint {target.x + radius * std::cos(newAngle), viewPoint.y, target.z + radius * std::sin(newAngle)};
    set(newViewPoint, target, unitY);
}

Camera Camera::rotatedAroundTarget(double degrees) const {
    auto newCamera = *this;
    newCamera.rotateAroundTarget(degrees);
    return newCamera;
}

const Point3D& Camera::getViewPoint() const {
    return viewPoint;
}

const Point3D& Camera::getTarget() const {
    return target;
}

const Vector3D& Camera::getUnitX() const {
    return unitX;
}

const Vector3D& Camera::getUnitY() const {
    return unitY;
}

const Vector3D& Camera::getUnitZ() const {
    return unitZ;
}

}
