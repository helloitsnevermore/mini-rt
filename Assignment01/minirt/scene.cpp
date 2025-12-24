#include "scene.h"
#include "ray.h"
#include "scene_loader.h"

#include <limits>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <cmath>
#include <map>

namespace minirt {

Scene::Scene() {}

void Scene::addSphere(const Sphere &sphere) {
    spheres.push_back(sphere);
}

void Scene::addLight(const PointLight &light) {
    lights.push_back(light);
}

void Scene::setAmbient(const Color &color) {
    ambientLight = color;
}

void Scene::setBackground(const Color &color){
    backgroundColor = color;
}

void Scene::setRecursionLimit(int limit) {
    limitOfRecursion = limit;
}

void Scene::setCamera(const Camera &camera) {
    this->camera = camera;
}

const Camera& Scene::getCamera() const {
    return camera;
}

void Scene::loadFromFile(const std::string &filename) {
    SceneLoader loader;
    loader.loadSceneFromFile(filename, *this);
}

// Returns closest object if there is an intersection or nullptr otherwise.
const Sphere* Scene::intersect(const Ray &ray, Point3D &closestIntersectionPoint) const {
    const Sphere *closestSphere = nullptr;
    double minDistance = std::numeric_limits<double>::max();
    for (const auto &sphere: spheres) {
        double distance = 0;
        if (sphere.intersect(ray, distance) && distance < minDistance) {
            closestSphere = &sphere;
            minDistance = distance;
        }
    }
    if (closestSphere) {
        closestIntersectionPoint = ray.fromOrigin(minDistance);
    }
    return closestSphere;
}

// Compute color for this ray in the scene.
Color Scene::illumination(const Ray &ray, int recursionStep) const {
    if (recursionStep >= limitOfRecursion) {
        return Color();
    }
    // Find an object for intersection.
    Point3D closestIntersectionPoint;
    auto closestObject = intersect(ray, closestIntersectionPoint);
    if (!closestObject) {
        return backgroundColor;
    }

    const auto &sphere = *closestObject;
    const auto &material = sphere.material;

    // Normal for the sphere and reflected ray.
    Vector3D normal = closestObject->normalTo(closestIntersectionPoint);
    Vector3D toViewer = -ray.direction;
    double cosThetaI = normal.dot(toViewer);
    Vector3D reflected = 2 * cosThetaI * normal - toViewer;

    // Add ambient light.
    Color color = ambientLight;

    // Add illumination from each light.
    for (const auto &light: lights) {
        // Check if the point on the object is illuminated by this light (not obscured by an obstacle).
        Vector3D toLight = light.position - closestIntersectionPoint;
        double distanceToLight = toLight.length();
        toLight = toLight.normalized();
        Point3D intersectionPoint;
        auto obstacle = intersect(Ray {closestIntersectionPoint, toLight}, intersectionPoint);
        if (obstacle) {
            // Check if the light is closer then the intersected object.
            double distanceToObstacle = (intersectionPoint - closestIntersectionPoint).length();
            if (distanceToObstacle > distanceToLight) {
                obstacle = nullptr;
            }
        }
        if (!obstacle) {
            // Apply coefficients of the body color to the intensity of the light source.
            color += material.shade(light.color, normal, reflected, toLight, toViewer);
        }
    }    

    double refractionCoeff = material.refractionCoeff;
    Vector3D refracted;
    // Check for refraction.
    if (refractionCoeff > 0) {
        double nu = 1.0 / material.refractionIndex; // assume refraction index 1.0 for air
        // Check if we hit object from inside.
        if (cosThetaI < 0) {
            nu = 1.0 /nu;
            normal = -normal;
            cosThetaI = -cosThetaI;
        }
        double cosThetaT = 1.0 - (1.0 - cosThetaI * cosThetaI) * (nu * nu);
        // Check for total internal reflection (no refraction).
        if (cosThetaT < 0) {
            refractionCoeff = 0.0;
        } else {
            cosThetaT = std::sqrt(cosThetaT);
            refracted = (cosThetaI * nu - cosThetaT) * normal - toViewer * nu;
        }
    }

    // Add refraction.
    if (refractionCoeff > 0) {
        Ray refractedRay {closestIntersectionPoint, refracted.normalized()};
        Color refractionColor = illumination(refractedRay, recursionStep + 1);
        color += refractionCoeff * refractionColor;
    }

    // Add reflection.
    double reflectionCoeff = 1.0 - refractionCoeff;
    if (reflectionCoeff > 0) {
        Ray reflectedRay {closestIntersectionPoint, reflected.normalized()};
        Color reflectionColor = illumination(reflectedRay, recursionStep + 1);
        color += reflectionCoeff * material.specularColor * reflectionColor;
    }

    return (color);
}

}
