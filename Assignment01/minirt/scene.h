#pragma once

#include "sphere.h"
#include "point_light.h"
#include "color.h"
#include "camera.h"

#include <vector>
#include <string>

namespace minirt {

class Ray;

class Scene  {
public:
    Scene();

    void addSphere(const Sphere &sphere);
    void addLight(const PointLight &light);        

    void setAmbient(const Color &color);
    void setBackground(const Color &color);
    void setRecursionLimit(int limit);

    void setCamera(const Camera &camera);
    const Camera& getCamera() const;

    void loadFromFile(const std::string &filename);

    // Returns closest object if there is an intersection or nullptr otherwise.
    const Sphere* intersect(const Ray &ray, Point3D &closestIntersectionPoint) const;

    // Compute color for this ray in the scene.
    Color illumination(const Ray &ray, int recursionStep = 0) const;    

private:
    std::vector<Sphere> spheres;
    std::vector<PointLight> lights;        

    Color ambientLight {0.1, 0.1, 0.1};
    Color backgroundColor {0};

    int limitOfRecursion {1};

    Camera camera;
};

}
