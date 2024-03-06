#include "arg_parse.hpp"

#include <cstring>
#include <iostream>

Args::Args()
    : generate_blue_noise_(false),
      use_opencl_(true),
      overwrite_file_(false),
      use_vulkan_(true),
      blue_noise_size_(32),
      threads_(4),
      output_filename_("output.png") {}

void Args::DisplayHelp() {
  std::cout << "[-h | --help] [-b <size> | --blue-noise <size>] [--usecl | "
               "--nousecl]\n"
               "  -h | --help\t\t\t\tDisplay this help text\n"
               "  -b <size> | --blue-noise <size>\tGenerate blue noise "
               "square with "
               "size\n"
               "  --usecl | --nousecl\t\t\tUse/Disable OpenCL (enabled by "
               "default)\n"
               "  -t <int> | --threads <int>\t\tUse CPU thread count when "
               "not using "
               "OpenCL\n"
               "  -o <filelname> | --output <filename>\tOutput filename to "
               "use\n"
               "  --overwrite\t\t\t\tEnable overwriting of file (default "
               "disabled)\n"
               "  --usevulkan | --nousevulkan\t\t\tUse/Disable Vulkan (enabled "
               "by default)\n";
}

bool Args::ParseArgs(int argc, char **argv) {
  --argc;
  ++argv;
  while (argc > 0) {
    if (std::strcmp(argv[0], "-h") == 0 ||
        std::strcmp(argv[0], "--help") == 0) {
      DisplayHelp();
      return true;
    } else if (std::strcmp(argv[0], "--usecl") == 0) {
      use_opencl_ = true;
    } else if (std::strcmp(argv[0], "--nousecl") == 0) {
      use_opencl_ = false;
    } else if (std::strcmp(argv[0], "--overwrite") == 0) {
      overwrite_file_ = true;
    } else if (argc > 1 && (std::strcmp(argv[0], "-b") == 0 ||
                            std::strcmp(argv[0], "--blue-noise") == 0)) {
      generate_blue_noise_ = true;
      blue_noise_size_ = std::strtoul(argv[1], nullptr, 10);
      if (blue_noise_size_ == 0) {
        std::cout << "ERROR: Failed to parse size for blue-noise" << std::endl;
        generate_blue_noise_ = false;
      }
      --argc;
      ++argv;
    } else if (argc > 1 && (std::strcmp(argv[0], "-t") == 0 ||
                            std::strcmp(argv[0], "--threads") == 0)) {
      threads_ = std::strtoul(argv[1], nullptr, 10);
      if (threads_ == 0) {
        std::cout << "ERROR: Failed to parse thread count, using 4 by "
                     "default"
                  << std::endl;
        threads_ = 4;
      }
      --argc;
      ++argv;
    } else if (argc > 1 && (std::strcmp(argv[0], "-o") == 0 ||
                            std::strcmp(argv[0], "--output") == 0)) {
      output_filename_ = std::string(argv[1]);
      --argc;
      ++argv;
    } else if (std::strcmp(argv[0], "--usevulkan") == 0) {
      use_vulkan_ = true;
    } else if (std::strcmp(argv[0], "--nousevulkan") == 0) {
      use_vulkan_ = false;
    } else {
      std::cout << "WARNING: Ignoring invalid input \"" << argv[0] << "\""
                << std::endl;
    }
    --argc;
    ++argv;
  }

  return false;
}
