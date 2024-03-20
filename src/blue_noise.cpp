#include "blue_noise.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>

#if DITHERING_OPENCL_ENABLED == 1
#include <CL/opencl.h>
#endif

#if DITHERING_VULKAN_ENABLED == 1
#include <vulkan/vulkan.h>

static std::vector<const char *> VK_EXTENSIONS = {};

#if VULKAN_VALIDATION == 1
const std::array<const char *, 1> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"};

static VKAPI_ATTR VkBool32 VKAPI_CALL fn_VULKAN_DEBUG_CALLBACK(
    VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT *p_callback_data, void *) {
  std::cerr << "Validation layer: " << p_callback_data->pMessage << std::endl;

  return VK_FALSE;
}
#endif  // VULKAN_VALIDATION == 1

struct QueueFamilyIndices {
  QueueFamilyIndices() : computeFamily() {}

  std::optional<uint32_t> computeFamily;

  bool isComplete() { return computeFamily.has_value(); }
};

QueueFamilyIndices find_queue_families(VkPhysicalDevice device) {
  QueueFamilyIndices indices;
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                           queue_families.data());

  for (uint32_t qf_idx = 0; qf_idx < queue_family_count; ++qf_idx) {
    if (queue_families[qf_idx].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      indices.computeFamily = qf_idx;
    }

    if (indices.isComplete()) {
      break;
    }
  }

  return indices;
}

#endif  // DITHERING_VULKAN_ENABLED == 1

#include "image.hpp"

image::Bl dither::blue_noise(int width, int height, int threads,
                             bool use_opencl, bool use_vulkan) {
#if DITHERING_VULKAN_ENABLED == 1
  if (use_vulkan) {
    // Try to use Vulkan.
#if VULKAN_VALIDATION == 1
    // Check for validation support.
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    bool validation_supported = true;

    for (const char *layer_name : VALIDATION_LAYERS) {
      bool layer_found = false;

      for (const auto &layer_props : available_layers) {
        if (std::strcmp(layer_name, layer_props.layerName) == 0) {
          layer_found = true;
          break;
        }
      }

      if (!layer_found) {
        validation_supported = false;
        break;
      }
    }

    if (!validation_supported) {
      std::clog << "WARNING: Validation requested but not supported, cannot "
                   "use Vulkan!\n";
      goto ENDOF_VULKAN;
    }

    VK_EXTENSIONS.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif  // VULKAN_VALIDATION == 1

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Blue Noise Generation";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    create_info.enabledExtensionCount = VK_EXTENSIONS.size();
    create_info.ppEnabledExtensionNames = VK_EXTENSIONS.data();

    VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
#if VULKAN_VALIDATION == 1
    create_info.enabledLayerCount = VALIDATION_LAYERS.size();
    create_info.ppEnabledLayerNames = VALIDATION_LAYERS.data();

    const auto populate_debug_info =
        [](VkDebugUtilsMessengerCreateInfoEXT *info) {
          info->sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
          info->messageSeverity =
              VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
              VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
              VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
          info->messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
          info->pfnUserCallback = fn_VULKAN_DEBUG_CALLBACK;
        };

    populate_debug_info(&debug_create_info);

    create_info.pNext = &debug_create_info;
#else
    create_info.enabledLayerCount = 0;
    create_info.pNext = nullptr;
#endif  // VULKAN_VALIDATION == 1

    VkInstance instance;
    if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
      std::clog << "WARNING: Failed to create Vulkan instance!\n";
      goto ENDOF_VULKAN;
    }
    utility::Cleanup cleanup_vk_instance(
        [](void *ptr) { vkDestroyInstance(*((VkInstance *)ptr), nullptr); },
        &instance);

#if VULKAN_VALIDATION == 1
    populate_debug_info(&debug_create_info);
    VkDebugUtilsMessengerEXT debug_messenger;

    auto create_debug_utils_messenger_func =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkCreateDebugUtilsMessengerEXT");
    if (create_debug_utils_messenger_func != nullptr &&
        create_debug_utils_messenger_func(instance, &debug_create_info, nullptr,
                                          &debug_messenger) != VK_SUCCESS) {
      std::clog << "WARNING: Failed to set up Vulkan debug messenger!\n";
      goto ENDOF_VULKAN;
    }
    utility::Cleanup cleanup_debug_messenger(
        [instance](void *ptr) {
          auto func =
              (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
                  instance, "vkDestroyDebugUtilsMessengerEXT");
          if (func != nullptr) {
            func(instance, *((VkDebugUtilsMessengerEXT *)ptr), nullptr);
          }
        },
        &debug_messenger);
#endif  // VULKAN_VALIDATION == 1
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0) {
      std::clog << "WARNING: No GPUs available with Vulkan support!\n";
      goto ENDOF_VULKAN;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    std::optional<VkPhysicalDevice> gpu_dev_discrete;
    std::optional<VkPhysicalDevice> gpu_dev_integrated;
    for (const auto &device : devices) {
      auto indices = find_queue_families(device);

      VkPhysicalDeviceProperties dev_props{};
      vkGetPhysicalDeviceProperties(device, &dev_props);

      if (indices.isComplete()) {
        if (dev_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
          gpu_dev_discrete = device;
        } else if (dev_props.deviceType ==
                   VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
          gpu_dev_integrated = device;
        }
      }
    }

    VkPhysicalDevice phys_device;
    if (gpu_dev_discrete.has_value()) {
      std::clog << "NOTICE: Found discrete GPU supporting Vulkan compute.\n";
      phys_device = gpu_dev_discrete.value();
    } else if (gpu_dev_integrated.has_value()) {
      std::clog << "NOTICE: Found integrated GPU supporting Vulkan compute.\n";
      phys_device = gpu_dev_integrated.value();
    } else {
      std::clog << "WARNING: No suitable GPUs found!\n";
      goto ENDOF_VULKAN;
    }

    auto indices = find_queue_families(phys_device);
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    std::unordered_set<uint32_t> unique_queue_families = {
        indices.computeFamily.value()};

    float queue_priority = 1.0F;
    for (uint32_t queue_family : unique_queue_families) {
      VkDeviceQueueCreateInfo queue_create_info{};
      queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queue_create_info.queueFamilyIndex = queue_family;
      queue_create_info.queueCount = 1;
      queue_create_info.pQueuePriorities = &queue_priority;
      queue_create_infos.push_back(queue_create_info);
    }

    VkPhysicalDeviceFeatures device_features{};

    VkDeviceCreateInfo dev_create_info{};
    dev_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    dev_create_info.queueCreateInfoCount = queue_create_infos.size();
    dev_create_info.pQueueCreateInfos = queue_create_infos.data();

    dev_create_info.pEnabledFeatures = &device_features;

    dev_create_info.enabledExtensionCount = 0;

#if VULKAN_VALIDATION == 1
    dev_create_info.enabledLayerCount = VALIDATION_LAYERS.size();
    dev_create_info.ppEnabledLayerNames = VALIDATION_LAYERS.data();
#else
    dev_create_info.enabledLayerCount = 0;
#endif

    VkDevice device;
    if (vkCreateDevice(phys_device, &dev_create_info, nullptr, &device) !=
        VK_SUCCESS) {
      std::clog << "WARNING: Failed to create VkDevice!\n";
      goto ENDOF_VULKAN;
    }
    utility::Cleanup device_cleanup(
        [](void *ptr) { vkDestroyDevice(*((VkDevice *)ptr), nullptr); },
        &device);

    VkQueue compute_queue;
    vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &compute_queue);

    std::array<VkDescriptorSetLayoutBinding, 4> compute_layout_bindings{};
    compute_layout_bindings[0].binding = 0;
    compute_layout_bindings[0].descriptorCount = 1;
    compute_layout_bindings[0].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    compute_layout_bindings[0].pImmutableSamplers = nullptr;
    compute_layout_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    compute_layout_bindings[1].binding = 1;
    compute_layout_bindings[1].descriptorCount = 1;
    compute_layout_bindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    compute_layout_bindings[1].pImmutableSamplers = nullptr;
    compute_layout_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    compute_layout_bindings[2].binding = 2;
    compute_layout_bindings[2].descriptorCount = 1;
    compute_layout_bindings[2].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    compute_layout_bindings[2].pImmutableSamplers = nullptr;
    compute_layout_bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    compute_layout_bindings[3].binding = 3;
    compute_layout_bindings[3].descriptorCount = 1;
    compute_layout_bindings[3].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    compute_layout_bindings[3].pImmutableSamplers = nullptr;
    compute_layout_bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = compute_layout_bindings.size();
    layout_info.pBindings = compute_layout_bindings.data();

    VkDescriptorSetLayout compute_desc_set_layout;
    if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr,
                                    &compute_desc_set_layout) != VK_SUCCESS) {
      std::clog << "WARNING: Failed to create compute descriptor set layout!\n";
      goto ENDOF_VULKAN;
    }
    utility::Cleanup compute_desc_set_layout_cleanup(
        [device](void *ptr) {
          vkDestroyDescriptorSetLayout(device, *((VkDescriptorSetLayout *)ptr),
                                       nullptr);
        },
        &compute_desc_set_layout);

    // Check and compile compute shader.
    {
      std::array<const char *, 3> filenames{
          "blue_noise.glsl", "src/blue_noise.glsl", "../src/blue_noise.glsl"};
      bool success = false;
      for (const auto filename : filenames) {
        std::ifstream ifs(filename);
        if (ifs.good()) {
          ifs.close();
          std::string command("glslc -fshader-stage=compute -o compute.spv ");
          command.append(filename);
          if (std::system(command.c_str()) != 0) {
            std::clog << "WARNING: Failed to compile " << filename << "!\n";
            goto ENDOF_VULKAN;
          } else {
            success = true;
            break;
          }
        }
      }
      if (!success) {
        std::clog << "WARNING: Could not find blue_noise.glsl!\n";
        goto ENDOF_VULKAN;
      }
    }

    // Load shader.
    std::vector<char> shader;
    {
      std::ifstream ifs("compute.spv");
      if (!ifs.good()) {
        std::clog << "WARNING: Failed to find compute.spv!\n";
        goto ENDOF_VULKAN;
      }
      ifs.seekg(0, std::ios_base::end);
      auto size = ifs.tellg();
      shader.resize(size);

      ifs.seekg(0);
      ifs.read(shader.data(), size);
      ifs.close();
    }

    // create compute pipeline.
    VkPipelineLayout compute_pipeline_layout;
    VkPipeline compute_pipeline;
    utility::Cleanup cleanup_pipeline_layout(utility::Cleanup::Nop{});
    utility::Cleanup cleanup_pipeline(utility::Cleanup::Nop{});
    {
      VkShaderModuleCreateInfo shader_module_create_info{};
      shader_module_create_info.sType =
          VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      shader_module_create_info.codeSize = shader.size();
      shader_module_create_info.pCode =
          reinterpret_cast<const uint32_t *>(shader.data());

      VkShaderModule compute_shader_module;
      if (vkCreateShaderModule(device, &shader_module_create_info, nullptr,
                               &compute_shader_module) != VK_SUCCESS) {
        std::clog << "WARNING: Failed to create shader module!\n";
        goto ENDOF_VULKAN;
      }

      utility::Cleanup cleanup_shader_module(
          [device](void *ptr) {
            vkDestroyShaderModule(device, *((VkShaderModule *)ptr), nullptr);
          },
          &compute_shader_module);

      VkPipelineShaderStageCreateInfo compute_shader_stage_info{};
      compute_shader_stage_info.sType =
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      compute_shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      compute_shader_stage_info.module = compute_shader_module;
      compute_shader_stage_info.pName = "main";

      VkPipelineLayoutCreateInfo pipeline_layout_info{};
      pipeline_layout_info.sType =
          VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipeline_layout_info.setLayoutCount = 1;
      pipeline_layout_info.pSetLayouts = &compute_desc_set_layout;

      if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr,
                                 &compute_pipeline_layout) != VK_SUCCESS) {
        std::clog << "WARNING: Failed to create compute pipeline layout!\n";
        goto ENDOF_VULKAN;
      }
      cleanup_pipeline_layout = utility::Cleanup(
          [device](void *ptr) {
            vkDestroyPipelineLayout(device, *((VkPipelineLayout *)ptr),
                                    nullptr);
          },
          &compute_pipeline_layout);

      VkComputePipelineCreateInfo pipeline_info{};
      pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      pipeline_info.layout = compute_pipeline_layout;
      pipeline_info.stage = compute_shader_stage_info;

      if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   nullptr, &compute_pipeline) != VK_SUCCESS) {
        std::clog << "WARNING: Failed to create compute pipeline!\n";
        goto ENDOF_VULKAN;
      }
      cleanup_pipeline = utility::Cleanup(
          [device](void *ptr) {
            vkDestroyPipeline(device, *((VkPipeline *)ptr), nullptr);
          },
          &compute_pipeline);
    }
  }
ENDOF_VULKAN:
  std::clog << "TODO: Remove this once Vulkan support is implemented.\n";
  return {};
#else
  std::clog << "WARNING: Not compiled with Vulkan support!\n";
#endif  // DITHERING_VULKAN_ENABLED == 1

#if DITHERING_OPENCL_ENABLED == 1
  if (use_opencl) {
    // try to use OpenCL
    do {
      cl_device_id device;
      cl_context context;
      cl_program program;
      cl_int err;

      cl_platform_id platform;

      int filter_size = (width + height) / 2;

      err = clGetPlatformIDs(1, &platform, nullptr);
      if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to identify a platform\n";
        break;
      }

      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
      if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to get a device\n";
        break;
      }

      context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

      {
        char buf[1024];
        std::ifstream program_file("src/blue_noise.cl");
        if (!program_file.good()) {
          std::cerr << "ERROR: Failed to read \"src/blue_noise.cl\" "
                       "(not found?)\n";
          break;
        }
        std::string program_string;
        while (program_file.good()) {
          program_file.read(buf, 1024);
          if (int read_count = program_file.gcount(); read_count > 0) {
            program_string.append(buf, read_count);
          }
        }

        const char *string_ptr = program_string.c_str();
        std::size_t program_size = program_string.size();
        program = clCreateProgramWithSource(
            context, 1, (const char **)&string_ptr, &program_size, &err);
        if (err != CL_SUCCESS) {
          std::cerr << "OpenCL: Failed to create the program\n";
          clReleaseContext(context);
          break;
        }

        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
          std::cerr << "OpenCL: Failed to build the program\n";

          std::size_t log_size;
          clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                nullptr, &log_size);
          std::unique_ptr<char[]> log = std::make_unique<char[]>(log_size + 1);
          log[log_size] = 0;
          clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                                log.get(), nullptr);
          std::cerr << log.get() << std::endl;

          clReleaseProgram(program);
          clReleaseContext(context);
          break;
        }
      }

      std::cout << "OpenCL: Initialized, trying cl_impl..." << std::endl;
      std::vector<unsigned int> result = internal::blue_noise_cl_impl(
          width, height, filter_size, context, device, program);

      clReleaseProgram(program);
      clReleaseContext(context);

      if (!result.empty()) {
        return internal::rangeToBl(result, width);
      }
      std::cout << "ERROR: Empty result\n";
    } while (false);
  }
#else
  std::clog << "WARNING: Not compiled with OpenCL support!\n";
#endif

  std::cout << "Vulkan/OpenCL: Failed to setup/use or is not enabled, using "
               "regular impl..."
            << std::endl;
  return internal::rangeToBl(internal::blue_noise_impl(width, height, threads),
                             width);
}

std::vector<unsigned int> dither::internal::blue_noise_impl(int width,
                                                            int height,
                                                            int threads) {
  int count = width * height;
  std::vector<float> filter_out;
  filter_out.resize(count);

  int pixel_count = count * 4 / 10;
  std::vector<bool> pbp = random_noise(count, count * 4 / 10);
  pbp.resize(count);

#ifndef NDEBUG
  printf("Inserting %d pixels into image of max count %d\n", pixel_count,
         count);
  // generate image from randomized pbp
  FILE *random_noise_image = fopen("random_noise.pbm", "w");
  fprintf(random_noise_image, "P1\n%d %d\n", width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      fprintf(random_noise_image, "%d ",
              pbp[utility::twoToOne(x, y, width, height)] ? 1 : 0);
    }
    fputc('\n', random_noise_image);
  }
  fclose(random_noise_image);
#endif

  // #ifndef NDEBUG
  int iterations = 0;
  // #endif

  int filter_size = (width + height) / 2;

  std::unique_ptr<std::vector<float>> precomputed =
      std::make_unique<std::vector<float>>(
          internal::precompute_gaussian(filter_size));

  internal::compute_filter(pbp, width, height, count, filter_size, filter_out,
                           precomputed.get(), threads);
#ifndef NDEBUG
  internal::write_filter(filter_out, width, "filter_out_start.pgm");
#endif
  std::cout << "Begin BinaryArray generation loop\n";
  while (true) {
#ifndef NDEBUG
    //        if(++iterations % 10 == 0) {
    printf("Iteration %d\n", ++iterations);
//        }
#endif
    // get filter values
    internal::compute_filter(pbp, width, height, count, filter_size, filter_out,
                             precomputed.get(), threads);

    // #ifndef NDEBUG
    //         for(int i = 0; i < count; ++i) {
    //             int x, y;
    //             std::tie(x, y) = internal::oneToTwo(i, width);
    //             printf("%d (%d, %d): %f\n", i, x, y, filter_out[i]);
    //         }
    // #endif

    int min, max;
    std::tie(min, max) = internal::filter_minmax(filter_out, pbp);

    // remove 1
    pbp[max] = false;

    // get filter values again
    internal::compute_filter(pbp, width, height, count, filter_size, filter_out,
                             precomputed.get(), threads);

    // get second buffer's min
    int second_min;
    std::tie(second_min, std::ignore) =
        internal::filter_minmax(filter_out, pbp);

    if (second_min == max) {
      pbp[max] = true;
      break;
    } else {
      pbp[second_min] = true;
    }

    if (iterations % 100 == 0) {
      // generate blue_noise image from pbp
#ifndef NDEBUG
      FILE *blue_noise_image = fopen("blue_noise.pbm", "w");
      fprintf(blue_noise_image, "P1\n%d %d\n", width, height);
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          fprintf(blue_noise_image, "%d ",
                  pbp[utility::twoToOne(x, y, width, height)] ? 1 : 0);
        }
        fputc('\n', blue_noise_image);
      }
      fclose(blue_noise_image);
#endif
    }
  }
  internal::compute_filter(pbp, width, height, count, filter_size, filter_out,
                           precomputed.get(), threads);
#ifndef NDEBUG
  internal::write_filter(filter_out, width, "filter_out_final.pgm");
#endif

#ifndef NDEBUG
  // generate blue_noise image from pbp
  FILE *blue_noise_image = fopen("blue_noise.pbm", "w");
  fprintf(blue_noise_image, "P1\n%d %d\n", width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      fprintf(blue_noise_image, "%d ",
              pbp[utility::twoToOne(x, y, width, height)] ? 1 : 0);
    }
    fputc('\n', blue_noise_image);
  }
  fclose(blue_noise_image);
#endif

  std::cout << "Generating dither_array...\n";
  std::vector<unsigned int> dither_array(count);
  int min, max;
  {
    std::vector<bool> pbp_copy(pbp);
    std::cout << "Ranking minority pixels...\n";
    for (unsigned int i = pixel_count; i-- > 0;) {
#ifndef NDEBUG
      std::cout << i << ' ';
#endif
      internal::compute_filter(pbp, width, height, count, filter_size,
                               filter_out, precomputed.get(), threads);
      std::tie(std::ignore, max) = internal::filter_minmax(filter_out, pbp);
      pbp[max] = false;
      dither_array[max] = i;
    }
    pbp = pbp_copy;
  }
  std::cout << "\nRanking remainder of first half of pixels...\n";
  for (unsigned int i = pixel_count; i < (unsigned int)((count + 1) / 2); ++i) {
#ifndef NDEBUG
    std::cout << i << ' ';
#endif
    internal::compute_filter(pbp, width, height, count, filter_size, filter_out,
                             precomputed.get(), threads);
    std::tie(min, std::ignore) = internal::filter_minmax(filter_out, pbp);
    pbp[min] = true;
    dither_array[min] = i;
  }
  std::cout << "\nRanking last half of pixels...\n";
  std::vector<bool> reversed_pbp(pbp);
  for (unsigned int i = (count + 1) / 2; i < (unsigned int)count; ++i) {
#ifndef NDEBUG
    std::cout << i << ' ';
#endif
    for (unsigned int i = 0; i < pbp.size(); ++i) {
      reversed_pbp[i] = !pbp[i];
    }
    internal::compute_filter(reversed_pbp, width, height, count, filter_size,
                             filter_out, precomputed.get(), threads);
    std::tie(std::ignore, max) = internal::filter_minmax(filter_out, pbp);
    pbp[max] = true;
    dither_array[max] = i;
  }

  return dither_array;
}

#if DITHERING_OPENCL_ENABLED == 1
std::vector<unsigned int> dither::internal::blue_noise_cl_impl(
    const int width, const int height, const int filter_size,
    cl_context context, cl_device_id device, cl_program program) {
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

  d_filter_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                count * sizeof(float), nullptr, nullptr);
  d_precomputed =
      clCreateBuffer(context, CL_MEM_READ_ONLY,
                     precomputed.size() * sizeof(float), nullptr, nullptr);
  d_pbp = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(int),
                         nullptr, nullptr);

  err = clEnqueueWriteBuffer(queue, d_precomputed, CL_TRUE, 0,
                             precomputed.size() * sizeof(float),
                             &precomputed[0], 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL: Failed to write to d_precomputed buffer\n";
    clReleaseMemObject(d_pbp);
    clReleaseMemObject(d_precomputed);
    clReleaseMemObject(d_filter_out);
    clReleaseCommandQueue(queue);
    return {};
  }

  kernel = clCreateKernel(program, "do_filter", &err);
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL: Failed to create kernel: ";
    switch (err) {
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

  if (clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_filter_out) != CL_SUCCESS) {
    std::cerr << "OpenCL: Failed to set kernel arg 0\n";
    clReleaseKernel(kernel);
    clReleaseMemObject(d_pbp);
    clReleaseMemObject(d_precomputed);
    clReleaseMemObject(d_filter_out);
    clReleaseCommandQueue(queue);
    return {};
  }
  if (clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_precomputed) != CL_SUCCESS) {
    std::cerr << "OpenCL: Failed to set kernel arg 1\n";
    clReleaseKernel(kernel);
    clReleaseMemObject(d_pbp);
    clReleaseMemObject(d_precomputed);
    clReleaseMemObject(d_filter_out);
    clReleaseCommandQueue(queue);
    return {};
  }
  if (clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_pbp) != CL_SUCCESS) {
    std::cerr << "OpenCL: Failed to set kernel arg 2\n";
    clReleaseKernel(kernel);
    clReleaseMemObject(d_pbp);
    clReleaseMemObject(d_precomputed);
    clReleaseMemObject(d_filter_out);
    clReleaseCommandQueue(queue);
    return {};
  }
  if (clSetKernelArg(kernel, 3, sizeof(int), &width) != CL_SUCCESS) {
    std::cerr << "OpenCL: Failed to set kernel arg 3\n";
    clReleaseKernel(kernel);
    clReleaseMemObject(d_pbp);
    clReleaseMemObject(d_precomputed);
    clReleaseMemObject(d_filter_out);
    clReleaseCommandQueue(queue);
    return {};
  }
  if (clSetKernelArg(kernel, 4, sizeof(int), &height) != CL_SUCCESS) {
    std::cerr << "OpenCL: Failed to set kernel arg 4\n";
    clReleaseKernel(kernel);
    clReleaseMemObject(d_pbp);
    clReleaseMemObject(d_precomputed);
    clReleaseMemObject(d_filter_out);
    clReleaseCommandQueue(queue);
    return {};
  }
  if (filter_size % 2 == 0) {
    int filter_size_odd = filter_size + 1;
    if (clSetKernelArg(kernel, 5, sizeof(int), &filter_size_odd) !=
        CL_SUCCESS) {
      std::cerr << "OpenCL: Failed to set kernel arg 4\n";
      clReleaseKernel(kernel);
      clReleaseMemObject(d_pbp);
      clReleaseMemObject(d_precomputed);
      clReleaseMemObject(d_filter_out);
      clReleaseCommandQueue(queue);
      return {};
    }
  } else {
    if (clSetKernelArg(kernel, 5, sizeof(int), &filter_size) != CL_SUCCESS) {
      std::cerr << "OpenCL: Failed to set kernel arg 4\n";
      clReleaseKernel(kernel);
      clReleaseMemObject(d_pbp);
      clReleaseMemObject(d_precomputed);
      clReleaseMemObject(d_filter_out);
      clReleaseCommandQueue(queue);
      return {};
    }
  }

  if (clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                               sizeof(std::size_t), &local_size,
                               nullptr) != CL_SUCCESS) {
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

  bool reversed_pbp = false;

  const auto get_filter = [&queue, &kernel, &global_size, &local_size,
                           &d_filter_out, &d_pbp, &pbp, &pbp_i, &count, &filter,
                           &err, &reversed_pbp]() -> bool {
    for (unsigned int i = 0; i < pbp.size(); ++i) {
      if (reversed_pbp) {
        pbp_i[i] = pbp[i] ? 0 : 1;
      } else {
        pbp_i[i] = pbp[i] ? 1 : 0;
      }
    }
    if (clEnqueueWriteBuffer(queue, d_pbp, CL_TRUE, 0, count * sizeof(int),
                             &pbp_i[0], 0, nullptr, nullptr) != CL_SUCCESS) {
      std::cerr << "OpenCL: Failed to write to d_pbp buffer\n";
      return false;
    }

    if (err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size,
                                     &local_size, 0, nullptr, nullptr);
        err != CL_SUCCESS) {
      std::cerr << "OpenCL: Failed to enqueue task: ";
      switch (err) {
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

    clEnqueueReadBuffer(queue, d_filter_out, CL_TRUE, 0, count * sizeof(float),
                        &filter[0], 0, nullptr, nullptr);

    return true;
  };

  {
#ifndef NDEBUG
    printf("Inserting %d pixels into image of max count %d\n", pixel_count,
           count);
    // generate image from randomized pbp
    FILE *random_noise_image = fopen("random_noise.pbm", "w");
    fprintf(random_noise_image, "P1\n%d %d\n", width, height);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        fprintf(random_noise_image, "%d ",
                pbp[utility::twoToOne(x, y, width, height)] ? 1 : 0);
      }
      fputc('\n', random_noise_image);
    }
    fclose(random_noise_image);
#endif
  }

  if (!get_filter()) {
    std::cerr << "OpenCL: Failed to execute do_filter (at start)\n";
    clReleaseKernel(kernel);
    clReleaseMemObject(d_pbp);
    clReleaseMemObject(d_precomputed);
    clReleaseMemObject(d_filter_out);
    clReleaseCommandQueue(queue);
    return {};
  } else {
#ifndef NDEBUG
    internal::write_filter(filter, width, "filter_out_start.pgm");
#endif
  }

  int iterations = 0;

  std::cout << "Begin BinaryArray generation loop\n";
  while (true) {
#ifndef NDEBUG
    printf("Iteration %d\n", ++iterations);
#endif

    if (!get_filter()) {
      std::cerr << "OpenCL: Failed to execute do_filter\n";
      break;
    }

    int min, max;
    std::tie(min, max) = internal::filter_minmax(filter, pbp);

    pbp[max] = false;

    if (!get_filter()) {
      std::cerr << "OpenCL: Failed to execute do_filter\n";
      break;
    }

    // get second buffer's min
    int second_min;
    std::tie(second_min, std::ignore) = internal::filter_minmax(filter, pbp);

    if (second_min == max) {
      pbp[max] = true;
      break;
    } else {
      pbp[second_min] = true;
    }

    if (iterations % 100 == 0) {
#ifndef NDEBUG
      std::cout << "max was " << max << ", second_min is " << second_min
                << std::endl;
      // generate blue_noise image from pbp
      FILE *blue_noise_image = fopen("blue_noise.pbm", "w");
      fprintf(blue_noise_image, "P1\n%d %d\n", width, height);
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          fprintf(blue_noise_image, "%d ",
                  pbp[utility::twoToOne(x, y, width, height)] ? 1 : 0);
        }
        fputc('\n', blue_noise_image);
      }
      fclose(blue_noise_image);
#endif
    }
  }

  if (!get_filter()) {
    std::cerr << "OpenCL: Failed to execute do_filter (at end)\n";
  } else {
#ifndef NDEBUG
    internal::write_filter(filter, width, "filter_out_final.pgm");
    FILE *blue_noise_image = fopen("blue_noise.pbm", "w");
    fprintf(blue_noise_image, "P1\n%d %d\n", width, height);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        fprintf(blue_noise_image, "%d ",
                pbp[utility::twoToOne(x, y, width, height)] ? 1 : 0);
      }
      fputc('\n', blue_noise_image);
    }
    fclose(blue_noise_image);
#endif
  }

#ifndef NDEBUG
  {
    image::Bl pbp_image = toBl(pbp, width);
    pbp_image.writeToFile(image::file_type::PNG, true, "debug_pbp_before.png");
  }
#endif

  std::cout << "Generating dither_array...\n";
#ifndef NDEBUG
  std::unordered_set<unsigned int> set;
#endif
  std::vector<unsigned int> dither_array(count, 0);
  int min, max;
  {
    std::vector<bool> pbp_copy(pbp);
    std::cout << "Ranking minority pixels...\n";
    for (unsigned int i = pixel_count; i-- > 0;) {
#ifndef NDEBUG
      std::cout << i << ' ';
#endif
      get_filter();
      std::tie(std::ignore, max) = internal::filter_minmax(filter, pbp);
      pbp.at(max) = false;
      dither_array.at(max) = i;
#ifndef NDEBUG
      if (set.find(max) != set.end()) {
        std::cout << "\nWARNING: Reusing index " << max << '\n';
      } else {
        set.insert(max);
      }
#endif
    }
    pbp = pbp_copy;
#ifndef NDEBUG
    image::Bl min_pixels = internal::rangeToBl(dither_array, width);
    min_pixels.writeToFile(image::file_type::PNG, true, "da_min_pixels.png");
#endif
  }
  std::cout << "\nRanking remainder of first half of pixels...\n";
  for (unsigned int i = pixel_count; i < (unsigned int)((count + 1) / 2); ++i) {
#ifndef NDEBUG
    std::cout << i << ' ';
#endif
    get_filter();
    std::tie(min, std::ignore) = internal::filter_minmax(filter, pbp);
    pbp.at(min) = true;
    dither_array.at(min) = i;
#ifndef NDEBUG
    if (set.find(min) != set.end()) {
      std::cout << "\nWARNING: Reusing index " << min << '\n';
    } else {
      set.insert(min);
    }
#endif
  }
#ifndef NDEBUG
  {
    image::Bl min_pixels = internal::rangeToBl(dither_array, width);
    min_pixels.writeToFile(image::file_type::PNG, true, "da_mid_pixels.png");
    get_filter();
    internal::write_filter(filter, width, "filter_mid.pgm");
    image::Bl pbp_image = toBl(pbp, width);
    pbp_image.writeToFile(image::file_type::PNG, true, "debug_pbp_mid.png");
  }
#endif
  std::cout << "\nRanking last half of pixels...\n";
  reversed_pbp = true;
  for (unsigned int i = (count + 1) / 2; i < (unsigned int)count; ++i) {
#ifndef NDEBUG
    std::cout << i << ' ';
#endif
    get_filter();
    std::tie(std::ignore, max) = internal::filter_minmax(filter, pbp);
    pbp.at(max) = true;
    dither_array.at(max) = i;
#ifndef NDEBUG
    if (set.find(max) != set.end()) {
      std::cout << "\nWARNING: Reusing index " << max << '\n';
    } else {
      set.insert(max);
    }
#endif
  }
  std::cout << std::endl;

#ifndef NDEBUG
  {
    get_filter();
    internal::write_filter(filter, width, "filter_after.pgm");
    image::Bl pbp_image = toBl(pbp, width);
    pbp_image.writeToFile(image::file_type::PNG, true, "debug_pbp_after.png");
  }
#endif

  clReleaseKernel(kernel);
  clReleaseMemObject(d_pbp);
  clReleaseMemObject(d_precomputed);
  clReleaseMemObject(d_filter_out);
  clReleaseCommandQueue(queue);
  return dither_array;
}
#endif
