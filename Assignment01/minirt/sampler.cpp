#include "sampler.h"

#include <random>

namespace minirt {

Sampler::Sampler() {}

// Generate random samples in [0, 1] square.
std::vector<Sample2D> Sampler::sampleRandom(int numOfSamples) const {
    std::vector<Sample2D> samples;
    std::default_random_engine re;
    std::uniform_real_distribution <double> dist(0, 1);
    for (int i = 0; i < numOfSamples; i++) {
        const auto x = dist(re);
        const auto y = dist(re);
        samples.push_back({x, y});
    }
    return samples;
}

}
