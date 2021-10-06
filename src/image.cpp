#include "image.hpp"

#include <cstdio>
#include <random>

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

    switch(type) {
    case file_type::PBM:
        file = fopen(filename, "w");
        fprintf(file, "P1\n%d %d", width, height);
        break;
    case file_type::PGM:
        file = fopen(filename, "wb");
        fprintf(file, "P5\n%d %d\n255", width, height);
        break;
    case file_type::PPM:
        file = fopen(filename, "wb");
        fprintf(file, "P6\n%d %d\n255", width, height);
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
