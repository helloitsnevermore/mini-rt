#pragma once

#include "color.h"

#include <vector>
#include <string>

namespace minirt {

class Image {
public:
    Image();
    Image(int width, int height);

    // Get image data as array of pixels.
    // Pixels are stored by columns starting from the bottom left corner of the image.
    Color* getData();
    const Color* getData() const;

    int getWidth() const;
    int getHeight() const;

    int getNumOfPixels() const;

    // Set color for the pixel with a coordinate x (by width) and y (by height).
    // Coordinates are measured from the bottom left corner of the image.
    void set(int x, int y, const Color &color);

    // Get color of the pixel with a coordinate x (by width) and y (by height).
    // Coordinates are measured from the bottom left corner of the image.
    const Color& get(int x, int y) const;

    void saveJPEG(const std::string &filename, int quality = 90);

private:
    std::vector<Color> data;
    int width;
    int height;
};

}
