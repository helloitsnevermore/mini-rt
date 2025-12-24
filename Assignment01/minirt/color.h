#pragma once

#include <istream>

namespace minirt {

struct Color {
    double red;
    double green;
    double blue;

    Color();
    Color(double r, double g, double b);
    explicit Color(double c);

    void operator +=(const Color &c);

    Color clamp(double min = 0, double max = 1);
};

Color operator +(const Color &c1, const Color &c2);
Color operator -(const Color &c1, const Color &c2);
Color operator *(const Color &c1, const Color &c2);
Color operator *(const Color &c, double coeff);
Color operator *(double coeff, const Color &c);
Color operator /(const Color &c, double coeff);

std::istream& operator >> (std::istream &in, Color &c);

}
