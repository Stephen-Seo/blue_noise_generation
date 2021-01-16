#ifndef BLUE_NOISE_HPP
#define BLUE_NOISE_HPP

#include <vector>
#include <tuple>
#include <cmath>
#include <functional>
#include <unordered_set>

namespace dither {

std::vector<bool> blue_noise(std::size_t width, std::size_t height, std::size_t threads = 1);

namespace internal {
    inline std::size_t twoToOne(std::size_t x, std::size_t y, std::size_t width) {
        return x + y * width;
    }

    inline std::tuple<std::size_t, std::size_t> oneToTwo(std::size_t i, std::size_t width) {
        return {i % width, i / width};
    }

    constexpr double mu_squared = 1.5 * 1.5;

    inline double gaussian(double x, double y) {
        return std::exp(-(x*x + y*y)/(2*mu_squared));
    }

    inline double filter(
            const std::vector<bool>& pbp,
            std::size_t x, std::size_t y,
            std::size_t width, std::size_t height) {
        double sum = 0.0;

        // Should be range -M/2 to M/2, but size_t cannot be negative, so range
        // is 0 to M.
        // p' = (M + x - (p - M/2)) % M = (3M/2 + x - p) % M
        // q' = (N + y - (q - M/2)) % N = (N + M/2 + y - q) % N
        for(std::size_t q = 0; q < width; ++q) {
            std::size_t q_prime = (height + width / 2 + y - q) % height;
            for(std::size_t p = 0; p < width; ++p) {
                std::size_t p_prime = (3 * width / 2 + x - p) % width;
                bool pbp_value = pbp[twoToOne(p_prime, q_prime, width)];
                if(pbp_value) {
                    sum += gaussian((double)p - width/2.0, (double)q - width/2.0);
                }
            }
        }

        return sum;
    }

    inline std::tuple<std::size_t, std::size_t> filter_minmax(const std::vector<double>& filter) {
        double min = std::numeric_limits<double>::infinity();
        double max = 0.0;
        std::size_t min_index = 0;
        std::size_t max_index = 0;

        for(std::size_t i = 0; i < filter.size(); ++i) {
            if(filter[i] < min) {
                min_index = i;
                min = filter[i];
            }
            if(filter[i] > max) {
                max_index = i;
                max = filter[i];
            }
        }

        return {min_index, max_index};
    }

    void recursive_apply_radius(
        std::size_t idx, std::size_t width,
        std::size_t height, std::size_t radius,
        const std::function<bool(std::size_t)>& fn);

    bool recursive_apply_radius_impl(
        std::size_t idx, std::size_t width,
        std::size_t height, std::size_t radius,
        const std::function<bool(std::size_t)>& fn,
        std::unordered_set<std::size_t>& visited);

    inline std::size_t get_one_or_zero(
            const std::vector<bool>& pbp, bool get_one,
            std::size_t idx, std::size_t width, std::size_t height) {
        std::size_t found_idx;
        bool found = false;
        for(std::size_t radius = 1; radius <= 12; ++radius) {
            recursive_apply_radius(
                idx, width, height, radius,
                [&found_idx, &found, &pbp, &get_one] (std::size_t idx) {
                    if((get_one && pbp[idx]) || (!get_one && !pbp[idx])) {
                        found_idx = idx;
                        found = true;
                        return true;
                    } else {
                        return false;
                    }
                });
            if(found) {
                return found_idx;
            }
        }
        return idx;
    }
} // namespace dither::internal

} // namespace dither

#endif
