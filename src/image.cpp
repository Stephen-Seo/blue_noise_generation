#include "image.hpp"

#include <cstdio>
#include <random>
#include <iostream>

#include <png.h>

image::Bl::Bl() :
data(),
width(0),
height(0)
{}

image::Bl::Bl(int width, int height) :
data(width * height),
width(width),
height(height)
{}

image::Bl::Bl(const std::vector<uint8_t> &data, int width) :
data(data),
width(width),
height(data.size() / width)
{}

image::Bl::Bl(std::vector<uint8_t> &&data, int width) :
data(std::move(data)),
width(width),
height(data.size() / width)
{}

image::Bl::Bl(const std::vector<float> &data, int width) :
    data{},
    width(width),
    height(data.size() / width)
{
    for(float gspixel : data) {
        this->data.push_back(static_cast<uint8_t>(255.0F * gspixel));
    }
}

void image::Bl::randomize() {
    if(!isValid()) {
        return;
    }

    std::default_random_engine re(std::random_device{}());
    std::uniform_int_distribution<unsigned int> dist;

    for(unsigned int i = 0; i < data.size(); ++i) {
        data[i] = i < data.size() / 2 ? 255 : 0;
    }

    for(unsigned int i = 0; i < data.size() - 1; ++i) {
        int ridx = dist(re, decltype(dist)::param_type{i+1, (unsigned int)data.size()-1});
        uint8_t temp = data[i];
        data[i] = data[ridx];
        data[ridx] = temp;
    }
}

int image::Bl::getSize() {
    return data.size();
}

uint8_t* image::Bl::getData() {
    if(!isValid()) {
        return nullptr;
    }
    return &data[0];
}

const uint8_t* image::Bl::getDataC() const {
    if(!isValid()) {
        return nullptr;
    }
    return &data[0];
}

bool image::Bl::canWriteFile(file_type type) {
    if(!isValid()) {
        return false;
    }
    switch(type) {
    case file_type::PBM:
    case file_type::PGM:
    case file_type::PPM:
    case file_type::PNG:
        return true;
    default:
        return false;
    }
}

bool image::Bl::writeToFile(file_type type, bool canOverwrite, const char *filename) {
    if(!isValid() || !canWriteFile(type)) {
        return false;
    }

    FILE *file = fopen(filename, "r");
    if(file && !canOverwrite) {
        fclose(file);
        return false;
    }

    if(file) {
        fclose(file);
    }

    if(type == file_type::PNG) {
        FILE *outfile = fopen(filename, "wb");
        if (outfile == nullptr) {
            return false;
        }
        const static auto pngErrorLFn = [] (png_structp /* unused */,
                                            png_const_charp message) {
            std::cerr << "WARNING [libpng]: " << message << std::endl;
        };
        const static auto pngWarnLFn = [] (png_structp /* unused */,
                                           png_const_charp message) {
            std::cerr << "ERROR [libpng]: " << message << std::endl;
        };

        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                                      nullptr,
                                                      pngErrorLFn,
                                                      pngWarnLFn);

        if (png_ptr == nullptr) {
            fclose(outfile);
            return false;
        }

        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (info_ptr == nullptr) {
            png_destroy_write_struct(&png_ptr, nullptr);
            fclose(outfile);
            return false;
        }

        if (setjmp(png_jmpbuf(png_ptr))) {
            png_destroy_write_struct(&png_ptr, &info_ptr);
            fclose(outfile);
            return false;
        }

        png_init_io(png_ptr, outfile);

        png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_GRAY,
                     PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                     PNG_FILTER_TYPE_DEFAULT);

        png_write_info(png_ptr, info_ptr);

        //png_set_filler(png_ptr, 0, PNG_FILLER_AFTER);

        for (unsigned int j = 0; j < this->data.size() / this->width; ++j) {
            unsigned char *dataPtr = &this->data.at(j * this->width);
            png_write_rows(png_ptr, &dataPtr, 1);
        }

        png_write_end(png_ptr, nullptr);

        png_destroy_write_struct(&png_ptr, &info_ptr);

        fclose(outfile);
        return true;
    }

    switch(type) {
    case file_type::PBM:
        file = fopen(filename, "w");
        fprintf(file, "P1\n%d %d", width, height);
        break;
    case file_type::PGM:
        file = fopen(filename, "wb");
        fprintf(file, "P5\n%d %d\n255\n", width, height);
        break;
    case file_type::PPM:
        file = fopen(filename, "wb");
        fprintf(file, "P6\n%d %d\n255\n", width, height);
        break;
    default:
        fclose(file);
        return false;
    }
    for(unsigned int i = 0; i < data.size(); ++i) {
        if(type == file_type::PBM && i % width == 0) {
            fprintf(file, "\n");
        }
        switch(type) {
        case file_type::PBM:
            fprintf(file, "%d ", data[i] == 0 ? 0 : 1);
            break;
        case file_type::PGM:
            //fprintf(file, "%c ", data[i]);
            fputc(data[i], file);
            break;
        case file_type::PPM:
            //fprintf(file, "%c %c %c ", data[i], data[i], data[i]);
            fputc(data[i], file);
            fputc(data[i], file);
            fputc(data[i], file);
            break;
        default:
            fclose(file);
            return false;
        }
    }

    fclose(file);
    return true;
}

bool image::Bl::writeToFile(file_type type, bool canOverwrite, const std::string &filename) {
    return writeToFile(type, canOverwrite, filename.c_str());
}

bool image::Bl::isValid() const {
    return width > 0 && height > 0 && data.size() > 0;
}
