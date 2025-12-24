#include "material.h"
#include "vector_3d.h"

#include <algorithm>
#include <cmath>

namespace minirt {

Material::Material() {}

Material::Material(const Color &color) :
    diffuseColor(color) {
}

Material::Material(const Color &diffuse, const Color &specular, double shininess) :
    diffuseColor(diffuse), specularColor(specular), shininess(shininess) {
}

Material::Material(const Color &color, double diffuseCoeff, double specularCoeff, double shininess) :
    diffuseColor(color * diffuseCoeff), specularColor(color * specularCoeff), shininess(shininess) {
}

void Material::setDiffuse(const Color &color) {
    diffuseColor = color;
}

void Material::setSpecular(const Color &color) {
    specularColor = color;
}

void Material::setDiffuseAndSpecular(const Color &color) {
    diffuseColor = color;
    specularColor = color;
}

void Material::setShininess(double shininess) {
    this->shininess = shininess;
}

void Material::makeTransparent(double refrCoeff, double refrIndex) {
    refractionCoeff = refrCoeff;
    refractionIndex = refrIndex;
}

Color Material::shade(const Color &lightColor, const Vector3D &normal, const Vector3D &reflected, const Vector3D &toLight, const Vector3D &toViewer) const {
    double diffuse = std::max(toLight.dot(normal), 0.0);
    double specular = 0.0;
    if (diffuse > 0.0 && shininess > 0.0) {
        specular = std::pow(std::max(reflected.dot(toViewer), 0.0), shininess);
    }
    return (diffuseColor * diffuse + specularColor * specular) * lightColor;
}

}
