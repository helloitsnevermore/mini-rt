#pragma once

#include <vector>

namespace minirt {

struct Sample2D {
    double x;
    double y;
};

class Sampler {
public:
    Sampler();

    // Generate random samples in [0, 1] square.
    std::vector<Sample2D> sampleRandom(int numOfSamples) const;
};

}
