#include "point_light.h"

namespace minirt {

PointLight::PointLight() {}

PointLight::PointLight(const Point3D &position, const Color &color) :
    position(position), color(color) {            
}

}
