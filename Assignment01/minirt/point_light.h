#pragma once

#include "point_3d.h"
#include "color.h"

namespace minirt {

class PointLight {
public:
	PointLight();
    PointLight(const Point3D &position, const Color &color = Color(1));

public:    
    Point3D position;
    Color color;
};

}
