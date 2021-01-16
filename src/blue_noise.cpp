#include "blue_noise.hpp"

#include <random>

std::vector<bool> dither::blue_noise(std::size_t width, std::size_t height) {
    std::size_t count = width * height;
    std::vector<std::size_t> dither_array;
    dither_array.resize(count);

    std::vector<bool> pbp; // Prototype Binary Pattern
    pbp.resize(count);

    std::default_random_engine re(std::random_device{}());
    std::uniform_int_distribution<std::size_t> dist(0, count - 1);

    // initialize pbp
    for(std::size_t i = 0; i < count; ++i) {
        if(i < count / 2) {
            pbp[i] = true;
        } else {
            pbp[i] = false;
        }
    }
    // randomize pbp
    for(std::size_t i = 0; i < count-1; ++i) {
        decltype(dist)::param_type range{i+1, count-1};
        std::size_t ridx = dist(re, range);
        // probably can't use std::swap since using std::vector<bool>
        bool temp = pbp[i];
        pbp[i] = pbp[ridx];
        pbp[ridx] = temp;
    }



    return {};
}
