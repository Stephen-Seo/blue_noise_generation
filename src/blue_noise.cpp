#include "blue_noise.hpp"

#include <random>
#include <cassert>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>

#include <CL/opencl.h>

#ifndef NDEBUG
# include <cstdio>
#endif

image::Bl dither::blue_noise(int width, int height, int threads, bool use_opencl) {

    bool using_opencl = false;

    if(use_opencl) {
        // try to use OpenCL
        do {
            cl_device_id device;
            cl_context context;
            cl_program program;
            cl_int err;

            cl_platform_id platform;

            int filter_size = (width + height) / 2;

            err = clGetPlatformIDs(1, &platform, nullptr);
            if(err != CL_SUCCESS) {
                std::cerr << "OpenCL: Failed to identify a platform\n";
                break;
            }

            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
            if(err != CL_SUCCESS) {
                std::cerr << "OpenCL: Failed to get a device\n";
                break;
            }

            context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

            {
                char buf[1024];
                std::ifstream program_file("src/blue_noise.cl");
                std::string program_string;
                while(program_file.good()) {
                    program_file.read(buf, 1024);
                    if(int read_count = program_file.gcount(); read_count > 0) {
                        program_string.append(buf, read_count);
                    }
                }

                const char *string_ptr = program_string.c_str();
                std::size_t program_size = program_string.size();
                program = clCreateProgramWithSource(context, 1, (const char**)&string_ptr, &program_size, &err);
                if(err != CL_SUCCESS) {
                    std::cerr << "OpenCL: Failed to create the program\n";
                    clReleaseContext(context);
                    break;
                }

                err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                if(err != CL_SUCCESS) {
                    std::cerr << "OpenCL: Failed to build the program\n";

                    std::size_t log_size;
                    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                    std::unique_ptr<char[]> log = std::make_unique<char[]>(log_size + 1);
                    log[log_size] = 0;
                    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.get(), nullptr);
                    std::cerr << log.get() << std::endl;

                    clReleaseProgram(program);
                    clReleaseContext(context);
                    break;
                }
            }

            std::cout << "OpenCL: Initialized, trying cl_impl..." << std::endl;
            std::vector<bool> result = internal::blue_noise_cl_impl(
                width, height, filter_size, context, device, program);

            clReleaseProgram(program);
            clReleaseContext(context);

            if(!result.empty()) {
                return internal::toBl(result, width);
            }
        } while (false);
    }

    if(!using_opencl) {
        std::cout << "OpenCL: Failed to setup/use or is not enabled, using regular impl..."
            << std::endl;
        return internal::toBl(internal::blue_noise_impl(width, height, threads), width);
    }

    return {};
}

image::Bl dither::blue_noise_grayscale(int width, int height, int threads) {
    int count = width * height;
    std::vector<float> filter_out;
    filter_out.resize(count);

    std::vector<float> image = internal::random_noise_grayscale(count);

    int iterations = 0;
    int filter_size = (width + height) / 2;
    std::vector<float> precomputed(internal::precompute_gaussian(filter_size));

    int min, max;
    float tempPixel;
    int prevmin = -1;
    int prevmax = -1;
    while(true) {
        printf("Iteration %d\n", iterations);

        internal::compute_filter_grayscale(image,
                                           width, height, count,
                                           filter_size, filter_out,
                                           &precomputed, threads);

        std::tie(min, max) = internal::filter_minmax(filter_out);
        printf("min == %4d, max == %4d\n", min, max);
        tempPixel = image[max];
        image[max] = image[min];
        image[min] = tempPixel;
        if(prevmin >= 0 && prevmax >= 0
                && (utility::dist(min, prevmin, width) < 1.5F
                    || utility::dist(max, prevmax, width) < 1.5F)) {
            break;
        }
        prevmin = min;
        prevmax = max;

//#ifndef NDEBUG
        if(iterations % 20 == 0) {
            std::string name;
            name.append("tempGrayscale");
            if(iterations < 10) {
                name.append("00");
            } else if(iterations < 100) {
                name.append("0");
            }
            name.append(std::to_string(iterations));
            name.append(".pgm");
            image::Bl(image, width).writeToFile(image::file_type::PGM, true, name);
        }
//#endif
        ++iterations;
    }

    // TODO

    return image::Bl(image, width);
}

std::vector<bool> dither::internal::blue_noise_impl(int width, int height, int threads) {
    int count = width * height;
    std::vector<float> filter_out;
    filter_out.resize(count);

    int pixel_count = count * 4 / 10;
    std::vector<bool> pbp = random_noise(count, count * 4 / 10);
    pbp.resize(count);

//#ifndef NDEBUG
    printf("Inserting %d pixels into image of max count %d\n", pixel_count, count);
    // generate image from randomized pbp
    FILE *random_noise_image = fopen("random_noise.pbm", "w");
    fprintf(random_noise_image, "P1\n%d %d\n", width, height);
    for(int y = 0; y < height; ++y) {
        for(int x = 0; x < width; ++x) {
            fprintf(random_noise_image, "%d ", pbp[utility::twoToOne(x, y, width, height)] ? 1 : 0);
        }
        fputc('\n', random_noise_image);
    }
    fclose(random_noise_image);
//#endif

//#ifndef NDEBUG
    int iterations = 0;
//#endif

    int filter_size = (width + height) / 2;

    std::unique_ptr<std::vector<float>> precomputed = std::make_unique<std::vector<float>>(internal::precompute_gaussian(filter_size));

    internal::compute_filter(pbp, width, height, count, filter_size,
            filter_out, precomputed.get(), threads);
    internal::write_filter(filter_out, width, "filter_out_start.pgm");
    while(true) {
//#ifndef NDEBUG
//        if(++iterations % 10 == 0) {
            printf("Iteration %d\n", ++iterations);
//        }
//#endif
        // get filter values
        internal::compute_filter(pbp, width, height, count, filter_size,
                filter_out, precomputed.get(), threads);

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
                filter_out, precomputed.get(), threads);

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

        if(utility::dist(max_one, second_min, width) < 1.5f) {
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
                    fprintf(blue_noise_image, "%d ", pbp[utility::twoToOne(x, y, width, height)] ? 1 : 0);
                }
                fputc('\n', blue_noise_image);
            }
            fclose(blue_noise_image);
        }
    }
    internal::compute_filter(pbp, width, height, count, filter_size,
            filter_out, precomputed.get(), threads);
    internal::write_filter(filter_out, width, "filter_out_final.pgm");

//#ifndef NDEBUG
    // generate blue_noise image from pbp
    FILE *blue_noise_image = fopen("blue_noise.pbm", "w");
    fprintf(blue_noise_image, "P1\n%d %d\n", width, height);
    for(int y = 0; y < height; ++y) {
        for(int x = 0; x < width; ++x) {
            fprintf(blue_noise_image, "%d ", pbp[utility::twoToOne(x, y, width, height)] ? 1 : 0);
        }
        fputc('\n', blue_noise_image);
    }
    fclose(blue_noise_image);
//#endif

    return pbp;
}

std::vector<bool> dither::internal::blue_noise_cl_impl(
        int width, int height, int filter_size, cl_context context, cl_device_id device, cl_program program) {
    cl_int err;
    cl_kernel kernel;
    cl_command_queue queue;
    cl_mem d_filter_out, d_precomputed, d_pbp;
    std::size_t global_size, local_size;

    std::vector<float> precomputed = precompute_gaussian(filter_size);

    int count = width * height;
    int pixel_count = count * 4 / 10;
    std::vector<bool> pbp = random_noise(count, pixel_count);
    std::vector<int> pbp_i(pbp.size());

    queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);

    d_filter_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, count * sizeof(float), nullptr, nullptr);
    d_precomputed = clCreateBuffer(context, CL_MEM_READ_ONLY, filter_size * filter_size * sizeof(float), nullptr, nullptr);
    d_pbp = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(int), nullptr, nullptr);

    err = clEnqueueWriteBuffer(queue, d_precomputed, CL_TRUE, 0, filter_size * filter_size * sizeof(float), &precomputed[0], 0, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to write to d_precomputed buffer\n";
        clReleaseMemObject(d_pbp);
        clReleaseMemObject(d_precomputed);
        clReleaseMemObject(d_filter_out);
        clReleaseCommandQueue(queue);
        return {};
    }

    /*
    err = clEnqueueWriteBuffer(queue, d_pbp, CL_TRUE, 0, count * sizeof(int), &pbp_i[0], 0, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to write to d_pbp buffer\n";
        clReleaseMemObject(d_pbp);
        clReleaseMemObject(d_precomputed);
        clReleaseMemObject(d_filter_out);
        clReleaseCommandQueue(queue);
        return {};
    }
    */

    kernel = clCreateKernel(program, "do_filter", &err);
    if(err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to create kernel: ";
        switch(err) {
        case CL_INVALID_PROGRAM:
            std::cerr << "invalid program\n";
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
            std::cerr << "invalid program executable\n";
            break;
        case CL_INVALID_KERNEL_NAME:
            std::cerr << "invalid kernel name\n";
            break;
        case CL_INVALID_KERNEL_DEFINITION:
            std::cerr << "invalid kernel definition\n";
            break;
        case CL_INVALID_VALUE:
            std::cerr << "invalid value\n";
            break;
        case CL_OUT_OF_RESOURCES:
            std::cerr << "out of resources\n";
            break;
        case CL_OUT_OF_HOST_MEMORY:
            std::cerr << "out of host memory\n";
            break;
        default:
            std::cerr << "unknown error\n";
            break;
        }
        clReleaseMemObject(d_pbp);
        clReleaseMemObject(d_precomputed);
        clReleaseMemObject(d_filter_out);
        clReleaseCommandQueue(queue);
        return {};
    }

    if(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_filter_out) != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to set kernel arg 0\n";
        clReleaseKernel(kernel);
        clReleaseMemObject(d_pbp);
        clReleaseMemObject(d_precomputed);
        clReleaseMemObject(d_filter_out);
        clReleaseCommandQueue(queue);
        return {};
    }
    if(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_precomputed) != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to set kernel arg 1\n";
        clReleaseKernel(kernel);
        clReleaseMemObject(d_pbp);
        clReleaseMemObject(d_precomputed);
        clReleaseMemObject(d_filter_out);
        clReleaseCommandQueue(queue);
        return {};
    }
    if(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_pbp) != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to set kernel arg 2\n";
        clReleaseKernel(kernel);
        clReleaseMemObject(d_pbp);
        clReleaseMemObject(d_precomputed);
        clReleaseMemObject(d_filter_out);
        clReleaseCommandQueue(queue);
        return {};
    }
    if(clSetKernelArg(kernel, 3, sizeof(int), &width) != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to set kernel arg 3\n";
        clReleaseKernel(kernel);
        clReleaseMemObject(d_pbp);
        clReleaseMemObject(d_precomputed);
        clReleaseMemObject(d_filter_out);
        clReleaseCommandQueue(queue);
        return {};
    }
    if(clSetKernelArg(kernel, 4, sizeof(int), &height) != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to set kernel arg 4\n";
        clReleaseKernel(kernel);
        clReleaseMemObject(d_pbp);
        clReleaseMemObject(d_precomputed);
        clReleaseMemObject(d_filter_out);
        clReleaseCommandQueue(queue);
        return {};
    }
    if(clSetKernelArg(kernel, 5, sizeof(int), &filter_size) != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to set kernel arg 4\n";
        clReleaseKernel(kernel);
        clReleaseMemObject(d_pbp);
        clReleaseMemObject(d_precomputed);
        clReleaseMemObject(d_filter_out);
        clReleaseCommandQueue(queue);
        return {};
    }

    if(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(std::size_t), &local_size, nullptr) != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to get work group size\n";
        clReleaseKernel(kernel);
        clReleaseMemObject(d_pbp);
        clReleaseMemObject(d_precomputed);
        clReleaseMemObject(d_filter_out);
        clReleaseCommandQueue(queue);
        return {};
    }
    global_size = (std::size_t)std::ceil(count / (float)local_size) * local_size;

    std::cout << "OpenCL: global = " << global_size << ", local = " << local_size
        << std::endl;

    std::vector<float> filter(count);

    const auto get_filter = [&queue, &kernel, &global_size, &local_size,
            &d_filter_out, &d_pbp, &pbp, &pbp_i, &count, &filter, &err] () -> bool {
        for(unsigned int i = 0; i < pbp.size(); ++i) {
            pbp_i[i] = pbp[i] ? 1 : 0;
        }
        if(clEnqueueWriteBuffer(queue, d_pbp, CL_TRUE, 0, count * sizeof(int), &pbp_i[0], 0, nullptr, nullptr) != CL_SUCCESS) {
            std::cerr << "OpenCL: Failed to write to d_pbp buffer\n";
            return false;
        }

        if(err = clEnqueueNDRangeKernel(
                queue, kernel, 1, nullptr, &global_size, &local_size,
                0, nullptr, nullptr); err != CL_SUCCESS) {
            std::cerr << "OpenCL: Failed to enqueue task: ";
            switch(err) {
            case CL_INVALID_PROGRAM_EXECUTABLE:
                std::cerr << "invalid program executable\n";
                break;
            case CL_INVALID_COMMAND_QUEUE:
                std::cerr << "invalid command queue\n";
                break;
            case CL_INVALID_KERNEL:
                std::cerr << "invalid kernel\n";
                break;
            case CL_INVALID_CONTEXT:
                std::cerr << "invalid context\n";
                break;
            case CL_INVALID_KERNEL_ARGS:
                std::cerr << "invalid kernel args\n";
                break;
            case CL_INVALID_WORK_DIMENSION:
                std::cerr << "invalid work dimension\n";
                break;
            case CL_INVALID_GLOBAL_WORK_SIZE:
                std::cerr << "invalid global work size\n";
                break;
            case CL_INVALID_GLOBAL_OFFSET:
                std::cerr << "invalid global offset\n";
                break;
            case CL_INVALID_WORK_GROUP_SIZE:
                std::cerr << "invalid work group size\n";
                break;
            case CL_INVALID_WORK_ITEM_SIZE:
                std::cerr << "invalid work item size\n";
                break;
            case CL_MISALIGNED_SUB_BUFFER_OFFSET:
                std::cerr << "misaligned sub buffer offset\n";
                break;
            default:
                std::cerr << "Unknown\n";
                break;
            }
            return false;
        }

        clFinish(queue);

        clEnqueueReadBuffer(queue, d_filter_out, CL_TRUE, 0, count * sizeof(float), &filter[0], 0, nullptr, nullptr);

        return true;
    };

    {
        printf("Inserting %d pixels into image of max count %d\n", pixel_count, count);
        // generate image from randomized pbp
        FILE *random_noise_image = fopen("random_noise.pbm", "w");
        fprintf(random_noise_image, "P1\n%d %d\n", width, height);
        for(int y = 0; y < height; ++y) {
            for(int x = 0; x < width; ++x) {
                fprintf(random_noise_image, "%d ", pbp[utility::twoToOne(x, y, width, height)] ? 1 : 0);
            }
            fputc('\n', random_noise_image);
        }
        fclose(random_noise_image);
    }

    if(!get_filter()) {
        std::cerr << "OpenCL: Failed to execute do_filter (at start)\n";
        clReleaseKernel(kernel);
        clReleaseMemObject(d_pbp);
        clReleaseMemObject(d_precomputed);
        clReleaseMemObject(d_filter_out);
        clReleaseCommandQueue(queue);
        return {};
    } else {
        internal::write_filter(filter, width, "filter_out_start.pgm");
    }

    int iterations = 0;

    while(true) {
        printf("Iteration %d\n", ++iterations);

        if(!get_filter()) {
            std::cerr << "OpenCL: Failed to execute do_filter\n";
            break;
        }

        int min, max, min_zero, max_one;
        std::tie(min, max) = internal::filter_minmax(filter);
        if(!pbp[max]) {
            max_one = internal::get_one_or_zero(pbp, true, max, width, height);
        } else {
            max_one = max;
        }
        if(!pbp[max_one]) {
            std::cerr << "ERROR: Failed to find pbp[max] one" << std::endl;
            break;
        }

        if(pbp[min]) {
            min_zero = internal::get_one_or_zero(pbp, false, min, width, height);
        } else {
            min_zero = min;
        }
        if(pbp[min_zero]) {
            std::cerr << "ERROR: Failed to find pbp[min] zero" << std::endl;
            break;
        }

        pbp[max_one] = false;

        if(!get_filter()) {
            std::cerr << "OpenCL: Failed to execute do_filter\n";
            break;
        }

        // get second buffer's min
        int second_min;
        std::tie(second_min, std::ignore) = internal::filter_minmax(filter);
        if(pbp[second_min]) {
            second_min = internal::get_one_or_zero(pbp, false, second_min, width, height);
            if(pbp[second_min]) {
                std::cerr << "ERROR: Failed to find pbp[second_min] zero" << std::endl;
                break;
            }
        }

        if(utility::dist(max_one, second_min, width) < 1.5f) {
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
                    fprintf(blue_noise_image, "%d ", pbp[utility::twoToOne(x, y, width, height)] ? 1 : 0);
                }
                fputc('\n', blue_noise_image);
            }
            fclose(blue_noise_image);
        }
    }

    if(!get_filter()) {
        std::cerr << "OpenCL: Failed to execute do_filter (at end)\n";
    } else {
        internal::write_filter(filter, width, "filter_out_final.pgm");
        FILE *blue_noise_image = fopen("blue_noise.pbm", "w");
        fprintf(blue_noise_image, "P1\n%d %d\n", width, height);
        for(int y = 0; y < height; ++y) {
            for(int x = 0; x < width; ++x) {
                fprintf(blue_noise_image, "%d ", pbp[utility::twoToOne(x, y, width, height)] ? 1 : 0);
            }
            fputc('\n', blue_noise_image);
        }
        fclose(blue_noise_image);
    }

    clReleaseKernel(kernel);
    clReleaseMemObject(d_pbp);
    clReleaseMemObject(d_precomputed);
    clReleaseMemObject(d_filter_out);
    clReleaseCommandQueue(queue);
    return pbp;
}
