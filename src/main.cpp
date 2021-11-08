#include "blue_noise.hpp"

#include <iostream>

int main(int argc, char **argv) {
//#ifndef NDEBUG
    std::cout << "Trying blue_noise..." << std::endl;
    image::Bl bl = dither::blue_noise(32, 32, 15, true);
    bl.writeToFile(image::file_type::PNG, true, "blueNoiseOut.png");
//#endif

    return 0;
}
