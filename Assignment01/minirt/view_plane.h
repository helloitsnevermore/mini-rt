#pragma once

#include "color.h"
#include "vector_3d.h"
#include "point_3d.h"

namespace minirt {

class Scene;
class Camera;

class ViewPlane {
public:
    ViewPlane();
    ViewPlane(int resolutionX, int resolutionY, double sizeX, double sizeY, double distance);

    // Compute color for the pixel with the given index, using scene's camera.
    Color computePixel(const Scene &scene, int x, int y, int numOfSamples = 1);   

    // Compute color for the pixel with the given index, using provided camera.
    Color computePixel(const Scene &scene, const Camera &camera, int x, int y, int numOfSamples = 1);

private:
    int resolutionX;
    int resolutionY;
    double sizeX;
    double sizeY;
    double distance; // distance to view plane from view point
};

}
