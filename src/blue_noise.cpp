#include "blue_noise.hpp"

#include <random>
#include <cassert>
#include <iostream>

#ifndef NDEBUG
# include <cstdio>
#endif

std::vector<bool> dither::blue_noise(int width, int height, int threads) {
    int count = width * height;
    std::vector<float> filter_out;
    filter_out.resize(count);

    std::vector<bool> pbp; // Prototype Binary Pattern
    pbp.resize(count);

    std::default_random_engine re(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, count - 1);

    const int pixel_count = count * 4 / 10;

    // initialize pbp
    for(int i = 0; i < count; ++i) {
        if(i < pixel_count) {
            pbp[i] = true;
        } else {
            pbp[i] = false;
        }
    }
    // randomize pbp
    for(int i = 0; i < count-1; ++i) {
        decltype(dist)::param_type range{i+1, count-1};
        int ridx = dist(re, range);
        // probably can't use std::swap since using std::vector<bool>
        bool temp = pbp[i];
        pbp[i] = pbp[ridx];
        pbp[ridx] = temp;
    }
//#ifndef NDEBUG
    printf("Inserting %d pixels into image of max count %d\n", pixel_count, count);
    // generate image from randomized pbp
    FILE *random_noise_image = fopen("random_noise.pbm", "w");
    fprintf(random_noise_image, "P1\n%d %d\n", width, height);
    for(int y = 0; y < height; ++y) {
        for(int x = 0; x < width; ++x) {
            fprintf(random_noise_image, "%d ", pbp[internal::twoToOne(x, y, width)] ? 1 : 0);
        }
        fputc('\n', random_noise_image);
    }
    fclose(random_noise_image);
//#endif

//#ifndef NDEBUG
    int iterations = 0;
//#endif

    int filter_size = (width + height) / 2;

    internal::compute_filter(pbp, width, height, count, filter_size,
            filter_out, threads);
    internal::write_filter(filter_out, width, "filter_out_start.pgm");
    while(true) {
//#ifndef NDEBUG
//        if(++iterations % 10 == 0) {
            printf("Iteration %d\n", ++iterations);
//        }
//#endif
        // get filter values
        internal::compute_filter(pbp, width, height, count, filter_size,
                filter_out, threads);

#ifndef NDEBUG
//        for(int i = 0; i < count; ++i) {
//            int x, y;
//            std::tie(x, y) = internal::oneToTwo(i, width);
//            printf("%d (%d, %d): %f\n", i, x, y, filter_out[i]);
//        }
#endif

        int min, max, min_zero, max_one;
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
        internal::compute_filter(pbp, width, height, count, filter_size,
                filter_out, threads);

        // get second buffer's min
        int second_min;
        std::tie(second_min, std::ignore) = internal::filter_minmax(filter_out);
        if(pbp[second_min]) {
            second_min = internal::get_one_or_zero(pbp, false, second_min, width, height);
            if(pbp[second_min]) {
                std::cerr << "ERROR: Failed to find pbp[second_min] zero" << std::endl;
                break;
            }
        }

        if(internal::dist(max_one, second_min, width) < 1.5f) {
            pbp[max_one] = true;
            break;
        } else {
            pbp[min_zero] = true;
        }

        if(iterations % 100 == 0) {
            // generate blue_noise image from pbp
            FILE *blue_noise_image = fopen("blue_noise.pbm", "w");
            fprintf(blue_noise_image, "P1\n%d %d\n", width, height);
            for(int y = 0; y < height; ++y) {
                for(int x = 0; x < width; ++x) {
                    fprintf(blue_noise_image, "%d ", pbp[internal::twoToOne(x, y, width)] ? 1 : 0);
                }
                fputc('\n', blue_noise_image);
            }
            fclose(blue_noise_image);
        }
    }
    internal::compute_filter(pbp, width, height, count, filter_size,
            filter_out, threads);
    internal::write_filter(filter_out, width, "filter_out_final.pgm");

//#ifndef NDEBUG
    // generate blue_noise image from pbp
    FILE *blue_noise_image = fopen("blue_noise.pbm", "w");
    fprintf(blue_noise_image, "P1\n%d %d\n", width, height);
    for(int y = 0; y < height; ++y) {
        for(int x = 0; x < width; ++x) {
            fprintf(blue_noise_image, "%d ", pbp[internal::twoToOne(x, y, width)] ? 1 : 0);
        }
        fputc('\n', blue_noise_image);
    }
    fclose(blue_noise_image);
//#endif

    return pbp;
}
