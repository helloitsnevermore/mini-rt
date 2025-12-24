#include "image.h"
#include "jpge.h"

namespace minirt {

Image::Image() :
    width(0), height(0) {
}

Image::Image(int width, int height) :
    width(width), height(height) {
    data.resize(width * height);
}

Color* Image::getData() {
    return data.data();
}

const Color* Image::getData() const {
    return data.data();
}

int Image::getWidth() const {
    return width;
}

int Image::getHeight() const {
    return height;
}

int Image::getNumOfPixels() const {
    return width * height;
}

void Image::set(int x, int y, const Color &color) {
    data[x * height + y] = color;
}

const Color& Image::get(int x, int y) const {
    return data[x * height + y];
}

void Image::saveJPEG(const std::string &filename, int quality) {
    std::vector<jpge::uint8> imgData(width * height * 3);
    for(int x = 0; x < width; x++)
    for(int y = 0; y < height; y++) {
        const auto &color = get(x, y);
        // Write image by rows: y is row number, x - column number.
        // Y index in array goes from the bottom of the image - we need to flip the image by Y.
        const int dIndex = ((height - y - 1) * width + x) * 3;
        imgData[dIndex] = static_cast<jpge::uint8>(255 * color.red);
        imgData[dIndex + 1] = static_cast<jpge::uint8>(255 * color.green);
        imgData[dIndex + 2] = static_cast<jpge::uint8>(255 * color.blue);
    }
    jpge::params params;
    params.m_quality = quality;
    jpge::compress_image_to_jpeg_file(filename.c_str(), width, height, 3, imgData.data(), params);
}

}
