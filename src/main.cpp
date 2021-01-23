#include "blue_noise.hpp"

#include <iostream>

int main(int argc, char **argv) {
//#ifndef NDEBUG
    std::cout << "Trying blue_noise..." << std::endl;
    dither::blue_noise(100, 100, 8);
//#endif

    return 0;
}
