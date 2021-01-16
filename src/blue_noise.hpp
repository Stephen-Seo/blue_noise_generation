#ifndef BLUE_NOISE_HPP
#define BLUE_NOISE_HPP

#include <vector>
#include <tuple>
#include <cmath>

namespace dither {

std::vector<bool> blue_noise(std::size_t width, std::size_t height);

namespace internal {
    inline std::size_t twoToOne(std::size_t x, std::size_t y, std::size_t width) {
        return x + y * width;
    }

    inline std::tuple<std::size_t, std::size_t> oneToTwo(std::size_t i, std::size_t width) {
        return {i % width, i / width};
    }

    constexpr double mu_squared = 1.5 * 1.5;

    inline double gaussian(double x, double y) {
        return std::exp(-std::sqrt(x*x + y*y)/(2*mu_squared));
    }

    inline double filter(
            const std::vector<bool>& pbp,
            std::size_t x, std::size_t y,
            std::size_t width) {
        double sum = 0.0;

        // Should be range -M/2 to M/2, but size_t cannot be negative, so range
        // is 0 to M.
        // p' = (M + x - (p - M/2)) % M = (3M/2 + x - p) % M
        // q' = (M + y - (q - M/2)) % M = (3M/2 + y - q) % M
        for(std::size_t q = 0; q <= width; ++q) {
            std::size_t q_prime = (3 * width / 2 + y - q) % width;
            for(std::size_t p = 0; p <= width; ++p) {
                std::size_t p_prime = (3 * width / 2 + x - p) % width;
                bool pbp_value = pbp[twoToOne(p_prime, q_prime, width)];
                if(pbp_value) {
                    sum += gaussian((double)p - width/2.0, (double)q - width/2.0);
                }
            }
        }

        return sum;
    }
} // namespace dither::internal

} // namespace dither

#endif
