#ifndef BLUE_NOISE_HPP
#define BLUE_NOISE_HPP

#include <vector>
#include <utility>
#include <cmath>
#include <functional>
#include <unordered_set>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstdio>
#include <queue>

namespace dither {

std::vector<bool> blue_noise(int width, int height, int threads = 1);

namespace internal {
    inline int twoToOne(int x, int y, int width) {
        return x + y * width;
    }

    inline std::pair<int, int> oneToTwo(int i, int width) {
        return {i % width, i / width};
    }

    constexpr float mu_squared = 1.5f * 1.5f;

    inline float gaussian(float x, float y) {
        return std::exp(-(x*x + y*y)/(2*mu_squared));
    }

    inline float filter(
            const std::vector<bool>& pbp,
            int x, int y,
            int width, int height, int filter_size) {
        float sum = 0.0f;

        // Should be range -M/2 to M/2, but size_t cannot be negative, so range
        // is 0 to M.
        // p' = (M + x - (p - M/2)) % M = (3M/2 + x - p) % M
        // q' = (N + y - (q - M/2)) % N = (N + M/2 + y - q) % N
        for(int q = 0; q < filter_size; ++q) {
            int q_prime = (height + filter_size / 2 + y - q) % height;
            for(int p = 0; p < filter_size; ++p) {
                int p_prime = (width + filter_size / 2 + x - p) % width;
                bool pbp_value = pbp[twoToOne(p_prime, q_prime, width)];
                if(pbp_value) {
                    sum += gaussian((float)p - filter_size/2.0f, (float)q - filter_size/2.0f);
                }
            }
        }

        return sum;
    }

    inline void compute_filter(
            const std::vector<bool> &pbp, int width, int height,
            int count, int filter_size, std::vector<float> &filter_out,
            int threads = 1) {
        if(threads == 1) {
            for(int y = 0; y < height; ++y) {
                for(int x = 0; x < width; ++x) {
                    filter_out[internal::twoToOne(x, y, width)] =
                        internal::filter(pbp, x, y, width, height, filter_size);
                }
            }
        } else {
            if(threads == 0) {
                threads = 10;
            }
            int active_count = 0;
            std::mutex cv_mutex;
            std::condition_variable cv;
            for(int i = 0; i < count; ++i) {
                {
                    std::unique_lock lock(cv_mutex);
                    active_count += 1;
                }
                std::thread t([] (int *ac, std::mutex *cvm,
                            std::condition_variable *cv, int i,
                            const std::vector<bool> *pbp, int width,
                            int height, int filter_size,
                            std::vector<float> *fout) {
                        int x, y;
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

    inline std::pair<int, int> filter_minmax(const std::vector<float>& filter) {
        float min = std::numeric_limits<float>::infinity();
        float max = 0.0f;
        int min_index = 0;
        int max_index = 0;

        for(std::vector<float>::size_type i = 0; i < filter.size(); ++i) {
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

    inline int get_one_or_zero(
            const std::vector<bool>& pbp, bool get_one,
            int idx, int width, int height) {
        std::queue<int> checking_indices;

        auto xy = oneToTwo(idx, width);
        int count = 0;
        int loops = 0;
        enum { D_DOWN = 0, D_LEFT = 1, D_UP = 2, D_RIGHT = 3 } dir = D_RIGHT;
        int next;

        while(true) {
            if(count == 0) {
                switch(dir) {
                case D_RIGHT:
                    xy.first = (xy.first + 1) % width;
                    ++loops;
                    count = loops * 2 - 1;
                    dir = D_DOWN;
                    break;
                case D_DOWN:
                    xy.first = (xy.first + width - 1) % width;
                    count = loops * 2 - 1;
                    dir = D_LEFT;
                    break;
                case D_LEFT:
                    xy.second = (xy.second + height - 1) % height;
                    count = loops * 2 - 1;
                    dir = D_UP;
                    break;
                case D_UP:
                    xy.first = (xy.first + 1) % width;
                    count = loops * 2 - 1;
                    dir = D_RIGHT;
                    break;
                }
            } else {
                switch(dir) {
                case D_DOWN:
                    xy.second = (xy.second + 1) % height;
                    --count;
                    break;
                case D_LEFT:
                    xy.first = (xy.first + width - 1) % width;
                    --count;
                    break;
                case D_UP:
                    xy.second = (xy.second + height - 1) % height;
                    --count;
                    break;
                case D_RIGHT:
                    xy.first = (xy.first + 1) % width;
                    --count;
                    break;
                }
            }
            next = twoToOne(xy.first, xy.second, width);
            if((get_one && pbp[next]) || (!get_one && !pbp[next])) {
                return next;
            }
        }
        return idx;
    }

    inline void write_filter(const std::vector<float> &filter, int width, const char *filename) {
        int min, max;
        std::tie(min, max) = filter_minmax(filter);

        printf("Writing to %s, min is %.3f, max is %.3f\n", filename, filter[min], filter[max]);
        FILE *filter_image = fopen(filename, "w");
        fprintf(filter_image, "P2\n%d %d\n65535\n", width, (int)filter.size() / width);
        for(std::vector<float>::size_type i = 0; i < filter.size(); ++i) {
            fprintf(filter_image, "%d ",
                (int)(((filter[i] - filter[min])
                        / (filter[max] - filter[min]))
                    * 65535.0f));
            if((i + 1) % width == 0) {
                fputc('\n', filter_image);
            }
        }
        fclose(filter_image);
    }

    inline float dist(int a, int b, int width) {
        auto axy = oneToTwo(a, width);
        auto bxy = oneToTwo(b, width);
        float dx = axy.first - bxy.first;
        float dy = axy.second - bxy.second;
        return std::sqrt(dx * dx + dy * dy);
    }
} // namespace dither::internal

} // namespace dither

#endif
