#ifndef BLUE_NOISE_HPP
#define BLUE_NOISE_HPP

#include <limits>
#include <vector>
#include <functional>
#include <unordered_set>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstdio>
#include <queue>
#include <random>
#include <cassert>
#include <stdexcept>
#include <iostream>

#include <sys/sysinfo.h>

#include <CL/opencl.h>

#include "utility.hpp"
#include "image.hpp"

namespace dither {

image::Bl blue_noise(int width, int height, int threads = 1, bool use_opencl = true);

namespace internal {
    std::vector<unsigned int> blue_noise_impl(int width, int height, int threads = 1);
    std::vector<unsigned int> blue_noise_cl_impl(
        const int width, const int height, const int filter_size,
        cl_context context, cl_device_id device, cl_program program);

    inline std::vector<bool> random_noise(int size, int subsize) {
        std::vector<bool> pbp(size);
        std::default_random_engine re(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, size - 1);

        // initialize pbp
        for(int i = 0; i < size; ++i) {
            if(i < subsize) {
                pbp[i] = true;
            } else {
                pbp[i] = false;
            }
        }
        // randomize pbp
        for(int i = 0; i < size-1; ++i) {
            decltype(dist)::param_type range{i+1, size-1};
            int ridx = dist(re, range);
            // probably can't use std::swap since using std::vector<bool>
            bool temp = pbp[i];
            pbp[i] = pbp[ridx];
            pbp[ridx] = temp;
        }

        return pbp;
    }

    constexpr float mu = 1.5F;
    constexpr float mu_squared = mu * mu;
    constexpr float double_mu_squared = 2.0F * mu * mu;

    inline float gaussian(float x, float y) {
        return std::exp(-(x*x + y*y)/(double_mu_squared));
    }

    inline std::vector<float> precompute_gaussian(int size) {
        std::vector<float> precomputed;
        if (size % 2 == 0) {
            ++size;
        }
        precomputed.reserve(size * size);

        for(int i = 0; i < size * size; ++i) {
            auto xy = utility::oneToTwo(i, size);
            precomputed.push_back(gaussian(
                xy.first - (size / 2),
                xy.second - (size / 2)));
        }

        return precomputed;
    }

    inline float filter(
            const std::vector<bool>& pbp,
            int x, int y,
            int width, int height, int filter_size) {
        float sum = 0.0f;

        if (filter_size % 2 == 0) {
            ++filter_size;
        }

        // Should be range -M/2 to M/2, but size_t cannot be negative, so range
        // is 0 to M.
        // p' = (M + x - (p - M/2)) % M = (3M/2 + x - p) % M
        // q' = (N + y - (q - M/2)) % N = (N + M/2 + y - q) % N
        for(int q = 0; q < filter_size; ++q) {
            int q_prime = (height - filter_size / 2 + y + q) % height;
            for(int p = 0; p < filter_size; ++p) {
                int p_prime = (width - filter_size / 2 + x + p) % width;
                if(pbp[utility::twoToOne(p_prime, q_prime, width, height)]) {
                    sum += gaussian(p - filter_size/2,
                                    q - filter_size/2);
                }
            }
        }

        return sum;
    }

    inline float filter_with_precomputed(
            const std::vector<bool>& pbp,
            int x, int y,
            int width, int height, int filter_size,
            const std::vector<float> &precomputed) {
        float sum = 0.0f;

        if (filter_size % 2 == 0) {
            ++filter_size;
        }

        for(int q = 0; q < filter_size; ++q) {
            int q_prime = (height - filter_size / 2 + y + q) % height;
            for(int p = 0; p < filter_size; ++p) {
                int p_prime = (width - filter_size / 2 + x + p) % width;
                if(pbp[utility::twoToOne(p_prime, q_prime, width, height)]) {
                    sum += precomputed[utility::twoToOne(p, q, filter_size, filter_size)];
                }
            }
        }

        return sum;
    }

    inline void compute_filter(
            const std::vector<bool> &pbp, int width, int height,
            int count, int filter_size, std::vector<float> &filter_out,
            const std::vector<float> *precomputed = nullptr,
            int threads = 1) {
        if(threads == 1) {
            if(precomputed) {
                for(int y = 0; y < height; ++y) {
                    for(int x = 0; x < width; ++x) {
                        filter_out[utility::twoToOne(x, y, width, height)] =
                            internal::filter_with_precomputed(
                                pbp, x, y, width, height, filter_size, *precomputed);
                    }
                }
            } else {
                for(int y = 0; y < height; ++y) {
                    for(int x = 0; x < width; ++x) {
                        filter_out[utility::twoToOne(x, y, width, height)] =
                            internal::filter(pbp, x, y, width, height, filter_size);
                    }
                }
            }
        } else {
            if(threads == 0) {
                threads = 10;
            }
            int active_count = 0;
            std::mutex cv_mutex;
            std::condition_variable cv;
            if(precomputed) {
                for(int i = 0; i < count; ++i) {
                    {
                        std::unique_lock lock(cv_mutex);
                        active_count += 1;
                    }
                    std::thread t([] (int *ac, std::mutex *cvm,
                                std::condition_variable *cv, int i,
                                const std::vector<bool> *pbp, int width,
                                int height, int filter_size,
                                std::vector<float> *fout,
                                const std::vector<float> *precomputed) {
                            int x, y;
                            std::tie(x, y) = utility::oneToTwo(i, width);
                            (*fout)[i] = internal::filter_with_precomputed(
                                *pbp, x, y, width, height, filter_size, *precomputed);
                            std::unique_lock lock(*cvm);
                            *ac -= 1;
                            cv->notify_all();
                        },
                        &active_count, &cv_mutex, &cv, i, &pbp, width, height,
                        filter_size, &filter_out, precomputed);
                    t.detach();

                    std::unique_lock lock(cv_mutex);
                    while(active_count >= threads) {
                        cv.wait_for(lock, std::chrono::seconds(1));
                    }
                }
            } else {
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
                            std::tie(x, y) = utility::oneToTwo(i, width);
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
                        cv.wait_for(lock, std::chrono::seconds(1));
                    }
                }
            }
            std::unique_lock lock(cv_mutex);
            while(active_count > 0) {
                cv.wait_for(lock, std::chrono::seconds(1));
            }
        }

    }

    inline std::pair<int, int> filter_minmax(const std::vector<float> &filter,
                                             std::vector<bool> pbp) {
        // ensure minority pixel is "true"
        unsigned int count = 0;
        for (bool value : pbp) {
            if(value) {
                ++count;
            }
        }
        if (count * 2 >= pbp.size()) {
            //std::cout << "MINMAX flip\n"; // DEBUG
            for (unsigned int i = 0; i < pbp.size(); ++i) {
                pbp[i] = !pbp[i];
            }
        }

        float min = std::numeric_limits<float>::infinity();
        float max = -std::numeric_limits<float>::infinity();
        int min_index = -1;
        int max_index = -1;

        for(std::vector<float>::size_type i = 0; i < filter.size(); ++i) {
            if(!pbp[i] && filter[i] < min) {
                min_index = i;
                min = filter[i];
            }
            if(pbp[i] && filter[i] > max) {
                max_index = i;
                max = filter[i];
            }
        }

        return {min_index, max_index};
    }

    inline std::pair<int, int> filter_abs_minmax(
            const std::vector<float> &filter) {
        float min = std::numeric_limits<float>::infinity();
        float max = -std::numeric_limits<float>::infinity();
        int min_index = -1;
        int max_index = -1;

        std::default_random_engine re(std::random_device{}());
        std::size_t startIdx = std::uniform_int_distribution<std::size_t>(0, filter.size() - 1)(re);

        for(std::vector<float>::size_type i = startIdx; i < filter.size(); ++i) {
            if(filter[i] < min) {
                min_index = i;
                min = filter[i];
            }
            if(filter[i] > max) {
                max_index = i;
                max = filter[i];
            }
        }
        for(std::vector<float>::size_type i = 0; i < startIdx; ++i) {
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

        auto xy = utility::oneToTwo(idx, width);
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
            next = utility::twoToOne(xy.first, xy.second, width, height);
            if((get_one && pbp[next]) || (!get_one && !pbp[next])) {
                return next;
            }
        }
        return idx;
    }

    inline void write_filter(const std::vector<float> &filter, int width, const char *filename) {
        int min, max;
        std::tie(min, max) = filter_abs_minmax(filter);

        printf("Writing to %s, min is %.3f, max is %.3f\n", filename, filter[min], filter[max]);
        FILE *filter_image = fopen(filename, "w");
        fprintf(filter_image, "P2\n%d %d\n255\n", width, (int)filter.size() / width);
        for(std::vector<float>::size_type i = 0; i < filter.size(); ++i) {
            fprintf(filter_image, "%d ",
                (int)(((filter[i] - filter[min])
                        / (filter[max] - filter[min]))
                    * 255.0f));
            if((i + 1) % width == 0) {
                fputc('\n', filter_image);
            }
        }
        fclose(filter_image);
    }

    inline image::Bl toBl(const std::vector<bool>& pbp, int width) {
        image::Bl bwImage(width, pbp.size() / width);
        assert((unsigned long)bwImage.getSize() >= pbp.size()
                && "New image::Bl size too small (pbp's size is not a multiple of width)");

        for(unsigned int i = 0; i < pbp.size(); ++i) {
            bwImage.getData()[i] = pbp[i] ? 255 : 0;
        }

        return bwImage;
    }

    inline image::Bl rangeToBl(const std::vector<unsigned int> &values, int width) {
        int min = std::numeric_limits<int>::max();
        int max = std::numeric_limits<int>::min();

        for (int value : values) {
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
        }

#ifndef NDEBUG
        std::cout << "rangeToBl: Got min == " << min << " and max == " << max << std::endl;
#endif

        max -= min;

        image::Bl grImage(width, values.size() / width);
        assert((unsigned long)grImage.getSize() >= values.size()
                && "New image::Bl size too small (values' size is not a multiple of width)");

        for(unsigned int i = 0; i < values.size(); ++i) {
            grImage.getData()[i] = ((float)((int)(values[i]) - min) / (float)max) * 255.0F;
        }

        return grImage;
    }

    inline std::pair<int, int> filter_minmax_in_range(int start, int width,
                                                   int height,
                                                   int range,
                                                   const std::vector<float> &vec) {
        float max = -std::numeric_limits<float>::infinity();
        float min = std::numeric_limits<float>::infinity();

        int maxIdx = -1;
        int minIdx = -1;

        auto startXY = utility::oneToTwo(start, width);
        for(int y = startXY.second - range / 2; y <= startXY.second + range / 2; ++y) {
            for(int x = startXY.first - range / 2; x <= startXY.first + range / 2; ++x) {
                int idx = utility::twoToOne(x, y, width, height);
                if(idx == start) {
                    continue;
                }

                if(vec[idx] < min) {
                    min = vec[idx];
                    minIdx = idx;
                }

                if(vec[idx] > max) {
                    max = vec[idx];
                    maxIdx = idx;
                }
            }
        }

        if(minIdx < 0) {
            throw std::runtime_error("Invalid minIdx value");
        } else if(maxIdx < 0) {
            throw std::runtime_error("Invalid maxIdx value");
        }
        return {minIdx, maxIdx};
    }
} // namespace dither::internal

} // namespace dither

#endif
