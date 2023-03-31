#include <cstdio>
#include <iostream>

#include "arg_parse.hpp"
#include "blue_noise.hpp"

int main(int argc, char **argv) {
  Args args;
  if (args.ParseArgs(argc, argv)) {
    return 0;
  }

  // validation
  if (args.generate_blue_noise_) {
    if (args.output_filename_.empty()) {
      std::cout << "ERROR: Cannot generate blue-noise, output filename "
                   "is not specified"
                << std::endl;
      Args::DisplayHelp();
      return 1;
    } else if (args.blue_noise_size_ < 16) {
      std::cout << "ERROR: blue-noise size is too small" << std::endl;
      Args::DisplayHelp();
      return 1;
    } else if (!args.overwrite_file_) {
      FILE *file = std::fopen(args.output_filename_.c_str(), "r");
      if (file) {
        std::fclose(file);
        std::cout << "ERROR: overwrite not specified, but filename exists"
                  << std::endl;
        Args::DisplayHelp();
        return 1;
      }
    }
  } else {
    std::cout << "ERROR: No operation specified\n";
    Args::DisplayHelp();
  }

  if (args.generate_blue_noise_) {
    std::cout << "Generating blue_noise..." << std::endl;
    image::Bl bl =
        dither::blue_noise(args.blue_noise_size_, args.blue_noise_size_,
                           args.threads_, args.use_opencl_);
    if (!bl.writeToFile(image::file_type::PNG, args.overwrite_file_,
                        args.output_filename_)) {
      std::cout << "ERROR: Failed to write blue-noise to file\n";
    }
  }

  return 0;
}
