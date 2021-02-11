#include "blue_noise.hpp"

#include <iostream>

int main(int argc, char **argv) {
//#ifndef NDEBUG
    std::cout << "Trying blue_noise..." << std::endl;
    image::Bl bl = dither::blue_noise(100, 100, 8, true);
    bl.writeToFile(image::file_type::PBM, true, "blueNoiseOut.pbm");
//#endif

    return 0;
}
