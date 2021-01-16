#include "blue_noise.hpp"

#include <random>
#include <cassert>
#include <iostream>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>

#ifndef NDEBUG
# include <cstdio>
#endif

void dither::internal::recursive_apply_radius(
        std::size_t idx, std::size_t width, std::size_t height,
        std::size_t radius, const std::function<bool(std::size_t)>& fn) {
    std::unordered_set<std::size_t> visited;
#ifndef NDEBUG
    if(recursive_apply_radius_impl(idx, width, height, radius, fn, visited)) {
        puts("recursive_apply_radius_impl found result");
    } else {
        puts("recursive_apply_radius_impl did NOT find result");
    }
#else
    recursive_apply_radius_impl(idx, width, height, radius, fn, visited);
#endif
}

bool dither::internal::recursive_apply_radius_impl(
        std::size_t idx, std::size_t width, std::size_t height,
        std::size_t radius, const std::function<bool(std::size_t)>& fn,
        std::unordered_set<std::size_t>& visited) {
    if(fn(idx)) {
        return true;
    }
    std::size_t x, y, temp;
    std::tie(x, y) = oneToTwo(idx, width);

    if(x + 1 < width) {
        temp = idx + 1;
        if(visited.find(temp) == visited.end()) {
            visited.insert(temp);
            if(recursive_apply_radius_impl(
                    temp, width, height, radius - 1, fn, visited)) {
                return true;
            }
        }
    } else {
        temp = twoToOne(0, y, width);
        if(visited.find(temp) == visited.end()) {
            visited.insert(temp);
            if(recursive_apply_radius_impl(
                    twoToOne(0, y, width),
                    width, height, radius - 1,
                    fn, visited)) {
                return true;
            }
        }
    }

    if(x > 0) {
        temp = idx - 1;
        if(visited.find(temp) == visited.end()) {
            visited.insert(temp);
            if(recursive_apply_radius_impl(
                    idx - 1, width, height, radius - 1, fn, visited)) {
                return true;
            }
        }
    } else {
        temp = twoToOne(width - 1, y, width);
        if(visited.find(temp) == visited.end()) {
            if(recursive_apply_radius_impl(
                    temp, width, height, radius - 1, fn, visited)) {
                return true;
            }
        }
    }

    if(y + 1 < height) {
        temp = idx + width;
        if(visited.find(temp) == visited.end()) {
            visited.insert(temp);
            if(recursive_apply_radius_impl(
                    temp, width, height, radius - 1, fn, visited)) {
                return true;
            }
        }
    } else {
        temp = twoToOne(x, 0, width);
        if(visited.find(temp) == visited.end()) {
            visited.insert(temp);
            if(recursive_apply_radius_impl(
                    temp, width, height, radius - 1, fn, visited)) {
                return true;
            }
        }
    }

    if(y > 0) {
        temp = idx - width;
        if(visited.find(temp) == visited.end()) {
            visited.insert(temp);
            if(recursive_apply_radius_impl(
                    temp, width, height, radius - 1, fn, visited)) {
                return true;
            }
        }
    } else {
        temp = twoToOne(x, height - 1, width);
        if(visited.find(temp) == visited.end()) {
            visited.insert(temp);
            if(recursive_apply_radius_impl(
                    temp, width, height, radius - 1, fn, visited)) {
                return true;
            }
        }
    }
    return false;
}


std::vector<bool> dither::blue_noise(std::size_t width, std::size_t height, std::size_t threads) {
    std::size_t count = width * height;
    std::vector<double> filter_out;
    filter_out.resize(count);

    std::vector<bool> pbp; // Prototype Binary Pattern
    pbp.resize(count);

    std::default_random_engine re(std::random_device{}());
    std::uniform_int_distribution<std::size_t> dist(0, count - 1);

    const std::size_t pixel_count = count * 4 / 10;

    // initialize pbp
    for(std::size_t i = 0; i < count; ++i) {
        if(i < pixel_count) {
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
//#ifndef NDEBUG
    printf("Inserting %ld pixels into image of max count %ld\n", pixel_count, count);
    // generate image from randomized pbp
    FILE *random_noise_image = fopen("random_noise.pbm", "w");
    fprintf(random_noise_image, "P1\n%ld %ld\n", width, height);
    for(std::size_t y = 0; y < height; ++y) {
        for(std::size_t x = 0; x < width; ++x) {
            fprintf(random_noise_image, "%d ", pbp[internal::twoToOne(x, y, width)] ? 1 : 0);
        }
        fputc('\n', random_noise_image);
    }
    fclose(random_noise_image);
//#endif

//#ifndef NDEBUG
    std::size_t iterations = 0;
//#endif

    while(true) {
//#ifndef NDEBUG
//        if(++iterations % 10 == 0) {
            printf("Iteration %ld\n", ++iterations);
//        }
//#endif
        // get filter values
        if(threads == 1) {
            for(std::size_t y = 0; y < height; ++y) {
                for(std::size_t x = 0; x < width; ++x) {
                    filter_out[internal::twoToOne(x, y, width)] =
                        internal::filter(pbp, x, y, width, height);
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
                            std::size_t height, std::vector<double> *fout) {
                        std::size_t x, y;
                        std::tie(x, y) = internal::oneToTwo(i, width);
                        (*fout)[i] = internal::filter(*pbp, x, y, width, height);
                        std::unique_lock lock(*cvm);
                        *ac -= 1;
                        cv->notify_all();
                    },
                    &active_count, &cv_mutex, &cv, i, &pbp, width, height, &filter_out);
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

#ifndef NDEBUG
//        for(std::size_t i = 0; i < count; ++i) {
//            std::size_t x, y;
//            std::tie(x, y) = internal::oneToTwo(i, width);
//            printf("%ld (%ld, %ld): %f\n", i, x, y, filter_out[i]);
//        }
#endif

        std::size_t min, max, min_zero, max_one;
        std::tie(min, max) = internal::filter_minmax(filter_out);
        if(!pbp[max]) {
            max_one = internal::get_one_or_zero(pbp, true, max, width, height);
#ifndef NDEBUG
            std::cout << "Post get_one(...)" << std::endl;
#endif
        } else {
            max_one = max;
        }
        if(!pbp[max_one]) {
            std::cerr << "ERROR: Failed to find pbp[max] one" << std::endl;
            break;
        }

        if(pbp[min]) {
            min_zero = internal::get_one_or_zero(pbp, false, min, width, height);
#ifndef NDEBUG
            std::cout << "Post get_zero(...)" << std::endl;
#endif
        } else {
            min_zero = min;
        }
        if(pbp[min_zero]) {
            std::cerr << "ERROR: Failed to find pbp[min] zero" << std::endl;
            break;
        }

        // remove 1
        pbp[max_one] = false;

        // get filter values again
        if(threads == 1) {
            for(std::size_t y = 0; y < height; ++y) {
                for(std::size_t x = 0; x < width; ++x) {
                    filter_out[internal::twoToOne(x, y, width)] =
                        internal::filter(pbp, x, y, width, height);
                }
            }
        } else {
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
                            std::size_t height, std::vector<double> *fout) {
                        std::size_t x, y;
                        std::tie(x, y) = internal::oneToTwo(i, width);
                        (*fout)[i] = internal::filter(*pbp, x, y, width, height);
                        std::unique_lock lock(*cvm);
                        *ac -= 1;
                        cv->notify_all();
                    },
                    &active_count, &cv_mutex, &cv, i, &pbp, width, height, &filter_out);
                t.detach();

                std::unique_lock lock(cv_mutex);
                while(active_count >= threads) {
#ifndef NDEBUG
//                    std::cout << "1, active_count = " << active_count
//                        << ", pre wait_for" << std::endl;
#endif
                    cv.wait_for(lock, std::chrono::seconds(1));
#ifndef NDEBUG
//                    std::cout << "1, active_count = " << active_count
//                        << ", post wait_for" << std::endl;
#endif
                }
            }
            std::unique_lock lock(cv_mutex);
            while(active_count > 0) {
                cv.wait_for(lock, std::chrono::seconds(1));
            }
        }

        // get second buffer's min
        std::size_t second_min;
        std::tie(second_min, std::ignore) = internal::filter_minmax(filter_out);
        if(pbp[second_min]) {
            second_min = internal::get_one_or_zero(pbp, false, second_min, width, height);
            if(pbp[second_min]) {
                std::cerr << "ERROR: Failed to find pbp[second_min] zero" << std::endl;
                break;
            }
        }

        if(min_zero != second_min) {
            pbp[max_one] = true;
            break;
        } else {
            pbp[min_zero] = true;
        }
    }

//#ifndef NDEBUG
    // generate blue_noise image from pbp
    FILE *blue_noise_image = fopen("blue_noise.pbm", "w");
    fprintf(blue_noise_image, "P1\n%ld %ld\n", width, height);
    for(std::size_t y = 0; y < height; ++y) {
        for(std::size_t x = 0; x < width; ++x) {
            fprintf(blue_noise_image, "%d ", pbp[internal::twoToOne(x, y, width)] ? 1 : 0);
        }
        fputc('\n', blue_noise_image);
    }
    fclose(blue_noise_image);
//#endif

    return pbp;
}
