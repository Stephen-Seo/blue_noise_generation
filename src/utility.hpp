#ifndef DITHERING_UTILITY_HPP
#define DITHERING_UTILITY_HPP

#include <utility>
#include <cmath>

namespace utility {
    inline int twoToOne(int x, int y, int width) {
        return x + y * width;
    }

    inline std::pair<int, int> oneToTwo(int i, int width) {
        return {i % width, i / width};
    }

    inline float dist(int a, int b, int width) {
        auto axy = utility::oneToTwo(a, width);
        auto bxy = utility::oneToTwo(b, width);
        float dx = axy.first - bxy.first;
        float dy = axy.second - bxy.second;
        return std::sqrt(dx * dx + dy * dy);
    }
}

#endif
