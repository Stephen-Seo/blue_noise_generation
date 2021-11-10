#include "blue_noise.hpp"

#include <iostream>

int main(int argc, char **argv) {
    std::cout << "Trying blue_noise..." << std::endl;
    image::Bl bl = dither::blue_noise(32, 32, 15, false);
    if(!bl.writeToFile(image::file_type::PNG, true, "blueNoiseOut.png")) {
        std::cout << "ERROR: Failed to write result to file\n";
        std::cout << "size is " << bl.getSize() << ", width is "
                  << bl.getWidth() << ", height is " << bl.getHeight()
                  << std::endl;
    }

    return 0;
}
