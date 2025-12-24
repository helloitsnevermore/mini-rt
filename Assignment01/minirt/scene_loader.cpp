#include "scene_loader.h"

#include <stdexcept>
#include <string>
#include <algorithm>

namespace minirt {

namespace {

std::string toLower(const std::string &str) {
    std::string newStr = str;
    std::transform(str.begin(), str.end(), newStr.begin(),
        [](unsigned char c){ return std::tolower(c); });
    return newStr;
}

std::string nextTag(std::istream &in) {
    std::string tag;
    in >> tag;
    return toLower(tag);
}

}

SceneLoader::SceneLoader() {}

void SceneLoader::loadSceneFromFile(const std::string &filename, Scene &scene) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open " + filename);
    }
    while (!in.eof()) {
        const auto tag = nextTag(in);
        if (tag.empty()) {
            continue;
        }
        if (tag == "sphere") {
            const auto sphere = loadSphere(in);
            scene.addSphere(sphere);
        } else if (tag == "light") {
            const auto light = loadLight(in);
            scene.addLight(light);
        } else if (tag == "camera") {
            const auto camera = loadCamera(in);
            scene.setCamera(camera);
        } else if (tag == "mat" || tag == "material") {
            std::string materialName;
            in >> materialName;
            materials[materialName] = loadMaterial(in);
        } else if (tag == "ambient") {
            Color color;
            in >> color;
            scene.setAmbient(color);
        } else if (tag == "background") {
            Color color;
            in >> color;
            scene.setBackground(color);
        } else if (tag == "recursion") {
            int limit;
            in >> limit;
            scene.setRecursionLimit(limit);
        } else {
            throw std::runtime_error("Unknown tag: " + tag);
        }
    }
}

Sphere SceneLoader::loadSphere(std::ifstream &in) {
    Point3D pos {0};
    double radius {1.0};
    Material material {Color {1.0}};
    Color color;
    while (!in.eof()) {
        const auto tag = nextTag(in);
        if (tag.empty()) {
            continue;
        }
        if (tag == "pos" || tag == "position") {
            in >> pos;
        } else if (tag == "rad" || tag == "radius") {
            in >> radius;
        } else if (tag == "mat" || tag == "material") {
            std::string materialName;
            in >> materialName;
            auto it = materials.find(materialName);
            if (it == materials.end()) {
                throw std::runtime_error("Unlnown material name: " + materialName);
            }
            material = it->second;
        } else if (tag == "color") {
            in >> color;
            material.setDiffuseAndSpecular(color);
        } else if (tag == "diffuse") {
            in >> color;
            material.setDiffuse(color);
        } else if (tag == "specular") {
            in >> color;
            material.setSpecular(color);
        } else if (tag == "shininess") {
            double shininess;
            in >> shininess;
            material.setShininess(shininess);
        } else if (tag == "transparent") {
            double refrCoeff, refrIndex;
            in >> refrCoeff >> refrIndex;
            material.makeTransparent(refrCoeff, refrIndex);
        } else if (tag == "end") {
            break;
        } else {
            throw std::runtime_error("Unknown sphere parameter: " + tag);
        }
    }
    return Sphere {pos, radius, material};
}

PointLight SceneLoader::loadLight(std::ifstream &in) {
    Point3D pos {0};
    Color color;
    while (!in.eof()) {
        const auto tag = nextTag(in);
        if (tag.empty()) {
            continue;
        }
        if (tag == "pos" || tag == "position") {
            in >> pos;
        } else if (tag == "color") {
            in >> color;
        } else if (tag == "end") {
            break;
        } else {
            throw std::runtime_error("Unknown light parameter: " + tag);
        }
    }
    return PointLight {pos, color};
}

Material SceneLoader::loadMaterial(std::ifstream &in) {
    Material material {Color {1.0}};
    Color color;
    while (!in.eof()) {
        const auto tag = nextTag(in);
        if (tag.empty()) {
            continue;
        }
        if (tag == "color") {
            in >> color;
            material.setDiffuseAndSpecular(color);
        } else if (tag == "diffuse") {
            in >> color;
            material.setDiffuse(color);
        } else if (tag == "specular") {
            in >> color;
            material.setSpecular(color);
        } else if (tag == "shininess") {
            double shininess;
            in >> shininess;
            material.setShininess(shininess);
        } else if (tag == "transparent") {
            double refrCoeff, refrIndex;
            in >> refrCoeff >> refrIndex;
            material.makeTransparent(refrCoeff, refrIndex);
        } else if (tag == "end") {
            break;
        } else {
            throw std::runtime_error("Unknown material parameter: " + tag);
        }
    }
    return material;
}

Camera SceneLoader::loadCamera(std::ifstream &in) {
    Point3D viewPoint;
    Point3D target {0};
    Vector3D up {0, 1, 0};
    while (!in.eof()) {
        const auto tag = nextTag(in);
        if (tag.empty()) {
            continue;
        }
        if (tag == "pos" || tag == "position") {
            in >> viewPoint;
        } else if (tag == "target") {
            in >> target;
        } else if (tag == "up") {
            in >> up;
        } else if (tag == "end") {
            break;
        } else {
            throw std::runtime_error("Unknown camera parameter: " + tag);
        }
    }
    return Camera {viewPoint, target, up};
}

}
