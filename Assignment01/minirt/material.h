#pragma once

#include "color.h"

namespace minirt {

class Vector3D;

class Material {
public:
    Material();
    // Set the same color for diffuse and specular.
    explicit Material(const Color &color);
    // Set different colors for diffuse and specular.
    Material(const Color &diffuse, const Color &specular, double shininess = 25);
    // Set specified amounts of the same color for diffuse and specular.
    Material(const Color &color, double diffuseCoeff, double specularCoeff, double shininess = 25);

    void setDiffuse(const Color &color);
    void setSpecular(const Color &color);
    void setDiffuseAndSpecular(const Color &color);
    void setShininess(double shininess);

    // Set amount of refraction and index of refraction.
    // Amount of refraction should be in [0, 1].
    void makeTransparent(double refrCoeff = 1.0, double refrIndex = 1.0);

    // Compute diffuse + specular shading.
    Color shade(const Color &lightColor, const Vector3D &normal, const Vector3D &reflected, const Vector3D &toLight, const Vector3D &toViewer) const;

public:
    Color diffuseColor {1, 1, 1};
    Color specularColor {1, 1, 1};
    double shininess {25};
    // Amount of refraction in [0, 1].
    // 0 - no refraction, 1 - full refraction (no reflection).
    double refractionCoeff {0};
    // Index of refraction (1 for air).
    double refractionIndex {1};
};

}
