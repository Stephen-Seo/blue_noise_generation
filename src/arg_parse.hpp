#ifndef DITHERING_ARG_PARSE_HPP_
#define DITHERING_ARG_PARSE_HPP_

#include <string>

struct Args {
    Args();

    static void DisplayHelp();

    /// Returns true if help was printed
    bool ParseArgs(int argc, char **argv);

    bool generate_blue_noise_;
    bool use_opencl_;
    bool overwrite_file_;
    unsigned int blue_noise_size_;
    unsigned int threads_;
    std::string output_filename_;
};

#endif
