#ifndef BLUE_NOISE_HPP
#define BLUE_NOISE_HPP

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

#include <CL/opencl.h>

#include "utility.hpp"
#include "image.hpp"

namespace dither {

image::Bl blue_noise(int width, int height, int threads = 1, bool use_opencl = true);

image::Bl blue_noise_grayscale(int width, int height, int threads = 1);

namespace internal {
    std::vector<bool> blue_noise_impl(int width, int height, int threads = 1);
    std::vector<bool> blue_noise_cl_impl(
        int width, int height, int filter_size,
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

    inline std::vector<float> random_noise_grayscale(unsigned int size) {
        std::vector<float> graynoise;
        graynoise.reserve(size);
        std::default_random_engine re(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0F, 1.0F);

        for(unsigned int i = 0; i < size; ++i) {
            graynoise.push_back(static_cast<float>(i) / static_cast<float>(size - 1));
            //graynoise[i] = dist(re);
        }
        for(unsigned int i = 0; i < size - 1; ++i) {
            std::uniform_int_distribution<unsigned int> range(i + 1, size - 1);
            unsigned int ridx = range(re);
            float temp = graynoise[i];
            graynoise[i] = graynoise[ridx];
            graynoise[ridx] = temp;
        }

        return graynoise;
    }

    constexpr float mu_squared = 1.5f * 1.5f;

    inline float gaussian(float x, float y) {
        return std::exp(-(x*x + y*y)/(2*mu_squared));
    }

    inline std::vector<float> precompute_gaussian(int size) {
        std::vector<float> precomputed;
        precomputed.reserve(size * size);

        for(int i = 0; i < size * size; ++i) {
            auto xy = utility::oneToTwo(i, size);
            precomputed.push_back(gaussian(
                (float)xy.first - size / 2.0f, (float)xy.second - size / 2.0f));
        }

        return precomputed;
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
                if(pbp[utility::twoToOne(p_prime, q_prime, width, height)]) {
                    sum += gaussian((float)p - filter_size/2.0f,
                                    (float)q - filter_size/2.0f);
                }
            }
        }

        return sum;
    }

    inline float filter_grayscale(
            const std::vector<float> &image,
            int x, int y,
            int width, int height, int filter_size) {
        float sum = 0.0F;
        for(int q = 0; q < filter_size; ++q) {
            int q_prime = (height + filter_size / 2 + y - q) % height;
            for(int p = 0; p < filter_size; ++p) {
                int p_prime = (width + filter_size / 2 + x - p) % width;
                sum += image[utility::twoToOne(p_prime, q_prime, width, height)]
                        * gaussian((float)p - filter_size/2.0F,
                                   (float)q - filter_size/2.0F);
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

        for(int q = 0; q < filter_size; ++q) {
            int q_prime = (height + filter_size / 2 + y - q) % height;
            for(int p = 0; p < filter_size; ++p) {
                int p_prime = (width + filter_size / 2 + x - p) % width;
                if(pbp[utility::twoToOne(p_prime, q_prime, width, height)]) {
                    sum += precomputed[utility::twoToOne(p, q, filter_size, filter_size)];
                }
            }
        }

        return sum;
    }

    inline float filter_with_precomputed_grayscale(
            const std::vector<float>& image,
            int x, int y,
            int width, int height, int filter_size,
            const std::vector<float> &precomputed) {
        float sum = 0.0F;

        for(int q = 0; q < filter_size; ++q) {
            int q_prime = (height + filter_size / 2 + y - q) % height;
            for(int p = 0; p < filter_size; ++p) {
                int p_prime = (width + filter_size / 2 + x - p) % width;
                sum += image[utility::twoToOne(p_prime, q_prime, width, height)]
                        * precomputed[utility::twoToOne(p, q, filter_size, filter_size)];
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

    inline void compute_filter_grayscale(
            const std::vector<float> &image, int width, int height,
            int count, int filter_size, std::vector<float> &filter_out,
            const std::vector<float> *precomputed = nullptr,
            int threads = 1) {
        if(precomputed) {
            for(int y = 0; y < height; ++y) {
                for(int x = 0; x < width; ++x) {
                    filter_out[utility::twoToOne(x, y, width, height)] =
                        internal::filter_with_precomputed_grayscale(
                            image,
                            x, y,
                            width, height,
                            filter_size,
                            *precomputed);
                }
            }
        } else {
            for(int y = 0; y < height; ++y) {
                for(int x = 0; x < width; ++x) {
                    filter_out[utility::twoToOne(x, y, width, height)] =
                        internal::filter_grayscale(image,
                                                   x, y,
                                                   width, height,
                                                   filter_size);
                }
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
        std::tie(min, max) = filter_minmax(filter);

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
            bwImage.getData()[i] = pbp[i] ? 1 : 0;
        }

        return bwImage;
    }
} // namespace dither::internal

} // namespace dither

#endif
