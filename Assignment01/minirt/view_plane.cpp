#include "view_plane.h"
#include "point_3d.h"
#include "scene.h"
#include "ray.h"
#include "sampler.h"
#include "camera.h"

#include <algorithm>

namespace minirt {

ViewPlane::ViewPlane(int resolutionX, int resolutionY, double sizeX, double sizeY, double distance) :
    resolutionX(resolutionX), resolutionY(resolutionY),
    sizeX(sizeX), sizeY(sizeY), distance(distance) {
}

Color ViewPlane::computePixel(const Scene &scene, int x, int y, int numOfSamples) {
    return computePixel(scene, scene.getCamera(), x, y, numOfSamples);
}

// Compute color for the pixel with the given index.
Color ViewPlane::computePixel(const Scene &scene, const Camera &camera, int x, int y, int numOfSamples) {
    Color color;
    std::vector<Sample2D> samples;
    if (numOfSamples == 1) {
        // One sample - use pixel's center.
        samples = {{0.5, 0.5}};
    } else {
        // Use random samples.
        Sampler sampler;
        samples = sampler.sampleRandom(numOfSamples);
    }
    const auto aspect = static_cast<double>(resolutionX) / resolutionY;
    for (const auto &sample: samples) {
        const auto sx = static_cast<double>(x) + sample.x;
        const auto sy = static_cast<double>(y) + sample.y;
        // A ray from the eye.
        const auto dx = sx * sizeX * aspect / resolutionX - sizeX * aspect / 2;
        const auto dy = sy * sizeY / resolutionY - sizeY / 2;
        const auto dz = distance;        
        color += scene.illumination(camera.rayFrom(dx, dy, dz));
    }
    return (color / numOfSamples).clamp();
}

}
