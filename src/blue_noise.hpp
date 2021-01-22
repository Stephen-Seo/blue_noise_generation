#ifndef BLUE_NOISE_HPP
#define BLUE_NOISE_HPP

#include <vector>
#include <tuple>
#include <cmath>
#include <functional>
#include <unordered_set>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>

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
            std::size_t width, std::size_t height, std::size_t filter_size) {
        double sum = 0.0;

        // Should be range -M/2 to M/2, but size_t cannot be negative, so range
        // is 0 to M.
        // p' = (M + x - (p - M/2)) % M = (3M/2 + x - p) % M
        // q' = (N + y - (q - M/2)) % N = (N + M/2 + y - q) % N
        for(std::size_t q = 0; q < filter_size; ++q) {
            std::size_t q_prime = (height + filter_size / 2 + y - q) % height;
            for(std::size_t p = 0; p < filter_size; ++p) {
                std::size_t p_prime = (width + filter_size / 2 + x - p) % width;
                bool pbp_value = pbp[twoToOne(p_prime, q_prime, width)];
                if(pbp_value) {
                    sum += gaussian((double)p - width/2.0, (double)q - width/2.0);
                }
            }
        }

        return sum;
    }

    inline void compute_filter(
            const std::vector<bool> &pbp, std::size_t width, std::size_t height,
            std::size_t count, std::size_t filter_size, std::vector<double> &filter_out,
            std::size_t threads = 1) {
        if(threads == 1) {
            for(std::size_t y = 0; y < height; ++y) {
                for(std::size_t x = 0; x < width; ++x) {
                    filter_out[internal::twoToOne(x, y, width)] =
                        internal::filter(pbp, x, y, width, height, filter_size);
                }
            }
        } else {
            if(threads == 0) {
                threads = 10;
            }
            std::size_t active_count = 0;
            std::mutex cv_mutex;
            std::condition_variable cv;
            for(std::size_t i = 0; i < count; ++i) {
                {
                    std::unique_lock lock(cv_mutex);
                    active_count += 1;
                }
                std::thread t([] (std::size_t *ac, std::mutex *cvm,
                            std::condition_variable *cv, std::size_t i,
                            const std::vector<bool> *pbp, std::size_t width,
                            std::size_t height, std::size_t filter_size,
                            std::vector<double> *fout) {
                        std::size_t x, y;
                        std::tie(x, y) = internal::oneToTwo(i, width);
                        (*fout)[i] = internal::filter(
                            *pbp, x, y, width, height, filter_size);
                        std::unique_lock lock(*cvm);
                        *ac -= 1;
                        cv->notify_all();
                    },
                    &active_count, &cv_mutex, &cv, i, &pbp, width, height,
                    filter_size, &filter_out);
                t.detach();

                std::unique_lock lock(cv_mutex);
                while(active_count >= threads) {
#ifndef NDEBUG
//                    std::cout << "0, active_count = " << active_count
//                        << ", pre wait_for" << std::endl;
#endif
                    cv.wait_for(lock, std::chrono::seconds(1));
#ifndef NDEBUG
//                    std::cout << "0, active_count = " << active_count
//                        << ", post wait_for" << std::endl;
#endif
                }
            }
            std::unique_lock lock(cv_mutex);
            while(active_count > 0) {
                cv.wait_for(lock, std::chrono::seconds(1));
            }
        }

    }

    inline std::tuple<std::size_t, std::size_t> filter_minmax(const std::vector<double>& filter) {
        double min = std::numeric_limits<double>::infinity();
        double max = 0.0;
        std::size_t min_index = 0;
        std::size_t max_index = 0;

        for(std::vector<double>::size_type i = 0; i < filter.size(); ++i) {
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
