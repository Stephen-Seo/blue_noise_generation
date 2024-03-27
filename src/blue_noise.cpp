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

dither::internal::QueueFamilyIndices::QueueFamilyIndices() : computeFamily() {}

bool dither::internal::QueueFamilyIndices::isComplete() {
  return computeFamily.has_value();
}

dither::internal::QueueFamilyIndices
dither::internal::vulkan_find_queue_families(VkPhysicalDevice device) {
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

std::optional<uint32_t> dither::internal::vulkan_find_memory_type(
    VkPhysicalDevice phys_dev, uint32_t t_filter, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);

  for (uint32_t idx = 0; idx < mem_props.memoryTypeCount; ++idx) {
    if ((t_filter & (1 << idx)) &&
        (mem_props.memoryTypes[idx].propertyFlags & props) == props) {
      return idx;
    }
  }

  return std::nullopt;
}

bool dither::internal::vulkan_create_buffer(
    VkDevice device, VkPhysicalDevice phys_dev, VkDeviceSize size,
    VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer &buf,
    VkDeviceMemory &buf_mem) {
  VkBufferCreateInfo buf_info{};
  buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buf_info.size = size;
  buf_info.usage = usage;
  buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &buf_info, nullptr, &buf) != VK_SUCCESS) {
    std::clog << "WARNING: Failed to create buffer!\n";
    buf = nullptr;
    return false;
  }

  VkMemoryRequirements mem_reqs;
  vkGetBufferMemoryRequirements(device, buf, &mem_reqs);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_reqs.size;

  auto mem_type =
      vulkan_find_memory_type(phys_dev, mem_reqs.memoryTypeBits, props);
  if (!mem_type.has_value()) {
    vkDestroyBuffer(device, buf, nullptr);
    buf = nullptr;
    return false;
  }
  alloc_info.memoryTypeIndex = mem_type.value();

  if (vkAllocateMemory(device, &alloc_info, nullptr, &buf_mem) != VK_SUCCESS) {
    std::clog << "WARNING: Failed to allocate buffer memory!\n";
    vkDestroyBuffer(device, buf, nullptr);
    buf = nullptr;
    return false;
  }

  vkBindBufferMemory(device, buf, buf_mem, 0);

  return true;
}

void dither::internal::vulkan_copy_buffer(VkDevice device,
                                          VkCommandPool command_pool,
                                          VkQueue queue, VkBuffer src_buf,
                                          VkBuffer dst_buf, VkDeviceSize size,
                                          VkDeviceSize src_offset,
                                          VkDeviceSize dst_offset) {
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandPool = command_pool;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer command_buf;
  vkAllocateCommandBuffers(device, &alloc_info, &command_buf);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(command_buf, &begin_info);

  VkBufferCopy copy_region{};
  copy_region.size = size;
  copy_region.srcOffset = src_offset;
  copy_region.dstOffset = dst_offset;
  vkCmdCopyBuffer(command_buf, src_buf, dst_buf, 1, &copy_region);

  vkEndCommandBuffer(command_buf);

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buf;

  vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);

  vkFreeCommandBuffers(device, command_pool, 1, &command_buf);
}

void dither::internal::vulkan_copy_buffer_pieces(
    VkDevice device, VkCommandPool command_pool, VkQueue queue,
    VkBuffer src_buf, VkBuffer dst_buf,
    const std::vector<std::tuple<VkDeviceSize, VkDeviceSize>> &pieces) {
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandPool = command_pool;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer command_buf;
  vkAllocateCommandBuffers(device, &alloc_info, &command_buf);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(command_buf, &begin_info);

  std::vector<VkBufferCopy> regions;
  for (auto tuple : pieces) {
    VkBufferCopy copy_region{};
    copy_region.size = std::get<0>(tuple);
    copy_region.srcOffset = std::get<1>(tuple);
    copy_region.dstOffset = std::get<1>(tuple);
    regions.push_back(copy_region);
  }
  vkCmdCopyBuffer(command_buf, src_buf, dst_buf, regions.size(),
                  regions.data());

  vkEndCommandBuffer(command_buf);

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buf;

  vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);

  vkFreeCommandBuffers(device, command_pool, 1, &command_buf);
}

void dither::internal::vulkan_flush_buffer(VkDevice device,
                                           VkDeviceMemory memory) {
  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.pNext = nullptr;
  range.memory = memory;
  range.offset = 0;
  range.size = VK_WHOLE_SIZE;

  if (vkFlushMappedMemoryRanges(device, 1, &range) != VK_SUCCESS) {
    std::clog << "WARNING: vulkan_flush_buffer failed!\n";
  }
}

void dither::internal::vulkan_flush_buffer_pieces(
    VkDevice device, const VkDeviceSize phys_atom_size, VkDeviceMemory memory,
    const std::vector<std::tuple<VkDeviceSize, VkDeviceSize>> &pieces) {
  std::vector<VkMappedMemoryRange> ranges;
  for (auto tuple : pieces) {
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.pNext = nullptr;
    range.memory = memory;
    range.offset = std::get<1>(tuple);
    range.size = std::get<0>(tuple);

    // TODO dynamically handle multiple pieces for more efficient flushes.
    // This may not be necessary if pieces is always size 1.

    if (range.offset % phys_atom_size != 0) {
      range.offset = (range.offset / phys_atom_size) * phys_atom_size;
    }

    if (range.size < phys_atom_size) {
      range.size = phys_atom_size;
    } else if (range.size % phys_atom_size != 0) {
      range.size = (range.size / phys_atom_size) * phys_atom_size;
    }

    ranges.push_back(range);
  }

  if (vkFlushMappedMemoryRanges(device, ranges.size(), ranges.data()) !=
      VK_SUCCESS) {
    std::clog << "WARNING: vulkan_flush_buffer failed!\n";
  }
}

void dither::internal::vulkan_invalidate_buffer(VkDevice device,
                                                VkDeviceMemory memory) {
  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.pNext = nullptr;
  range.memory = memory;
  range.offset = 0;
  range.size = VK_WHOLE_SIZE;

  if (vkInvalidateMappedMemoryRanges(device, 1, &range) != VK_SUCCESS) {
    std::clog << "WARNING: vulkan_invalidate_buffer failed!\n";
  }
}

std::vector<unsigned int> dither::internal::blue_noise_vulkan_impl(
    VkDevice device, VkPhysicalDevice phys_device,
    VkCommandBuffer command_buffer, VkCommandPool command_pool, VkQueue queue,
    VkBuffer pbp_buf, VkPipeline pipeline, VkPipelineLayout pipeline_layout,
    VkDescriptorSet descriptor_set, VkBuffer filter_out_buf,
    VkPipeline minmax_pipeline, VkPipelineLayout minmax_pipeline_layout,
    VkDescriptorSet minmax_descriptor_set, VkBuffer max_in_buf,
    VkBuffer min_in_buf, VkBuffer max_out_buf, VkBuffer min_out_buf,
    VkBuffer state_buf, const int width, const int height) {
  const int size = width * height;
  const int pixel_count = size * 4 / 10;
  const int local_size = 256;
  const std::size_t global_size =
      (std::size_t)std::ceil((float)size / (float)local_size);

  std::vector<bool> pbp = random_noise(size, pixel_count);
  bool reversed_pbp = false;

  VkBuffer staging_pbp_buffer;
  VkDeviceMemory staging_pbp_buffer_mem;
  void *pbp_mapped;
  if (!internal::vulkan_create_buffer(device, phys_device, size * sizeof(int),
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                      staging_pbp_buffer,
                                      staging_pbp_buffer_mem)) {
    std::clog << "get_filter ERROR: Failed to create staging pbp buffer!\n";
    return {};
  }
  utility::Cleanup cleanup_staging_pbp_buf(
      [device](void *ptr) {
        vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
      },
      &staging_pbp_buffer);
  utility::Cleanup cleanup_staging_pbp_buf_mem(
      [device](void *ptr) {
        vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
      },
      &staging_pbp_buffer_mem);
  vkMapMemory(device, staging_pbp_buffer_mem, 0, size * sizeof(int), 0,
              &pbp_mapped);
  utility::Cleanup cleanup_pbp_mapped(
      [device](void *ptr) { vkUnmapMemory(device, *((VkDeviceMemory *)ptr)); },
      &staging_pbp_buffer_mem);
  int *pbp_mapped_int = (int *)pbp_mapped;

  VkBuffer staging_filter_buffer;
  VkDeviceMemory staging_filter_buffer_mem;
  void *filter_mapped;
  if (!internal::vulkan_create_buffer(device, phys_device, size * sizeof(int),
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                      staging_filter_buffer,
                                      staging_filter_buffer_mem)) {
    std::clog << "get_filter ERROR: Failed to create staging pbp buffer!\n";
    return {};
  }
  utility::Cleanup cleanup_staging_filter_buf(
      [device](void *ptr) {
        vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
      },
      &staging_filter_buffer);
  utility::Cleanup cleanup_staging_filter_buf_mem(
      [device](void *ptr) {
        vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
      },
      &staging_filter_buffer_mem);
  vkMapMemory(device, staging_filter_buffer_mem, 0, size * sizeof(float), 0,
              &filter_mapped);
  utility::Cleanup cleanup_filter_mapped(
      [device](void *ptr) { vkUnmapMemory(device, *((VkDeviceMemory *)ptr)); },
      &staging_filter_buffer_mem);
  float *filter_mapped_float = (float *)filter_mapped;

  std::vector<std::size_t> changed_indices;

  VkDeviceSize phys_atom_size;
  {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys_device, &props);
    phys_atom_size = props.limits.nonCoherentAtomSize;
  }

  {
#ifndef NDEBUG
    printf("Inserting %d pixels into image of max count %d\n", pixel_count,
           size);
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

  if (!vulkan_get_filter(
          device, phys_atom_size, command_buffer, command_pool, queue, pbp_buf,
          pipeline, pipeline_layout, descriptor_set, filter_out_buf, size, pbp,
          reversed_pbp, global_size, pbp_mapped_int, staging_pbp_buffer,
          staging_pbp_buffer_mem, staging_filter_buffer_mem,
          staging_filter_buffer, nullptr)) {
    std::cerr << "Vulkan: Failed to execute get_filter at start!\n";
  } else {
#ifndef NDEBUG
    internal::write_filter(vulkan_buf_to_vec(filter_mapped_float, size), width,
                           "filter_out_start.pgm");
#endif
  }

#ifndef NDEBUG
  int iterations = 0;
#endif

  std::cout << "Begin BinaryArray generation loop\n";
  while (true) {
#ifndef NDEBUG
    printf("Iteration %d\n", ++iterations);
#endif

    if (!vulkan_get_filter(device, phys_atom_size, command_buffer, command_pool,
                           queue, pbp_buf, pipeline, pipeline_layout,
                           descriptor_set, filter_out_buf, size, pbp,
                           reversed_pbp, global_size, pbp_mapped_int,
                           staging_pbp_buffer, staging_pbp_buffer_mem,
                           staging_filter_buffer_mem, staging_filter_buffer,
                           &changed_indices)) {
      std::cerr << "Vulkan: Failed to execute do_filter\n";
      break;
    }

    int min, max;
    auto vulkan_minmax_opt = vulkan_minmax(
        device, phys_device, command_buffer, command_pool, queue,
        minmax_pipeline, minmax_pipeline_layout, minmax_descriptor_set,
        max_in_buf, min_in_buf, max_out_buf, min_out_buf, state_buf, size,
        filter_mapped_float, pbp);
    if (!vulkan_minmax_opt.has_value()) {
      std::cerr << "Vulkan: vulkan_minmax returned nullopt!\n";
      return {};
    }
    std::tie(min, max) = vulkan_minmax_opt.value();
#ifndef NDEBUG
    std::cout << "vulkan_minmax: " << min << ", " << max << '\n';
    {
      int temp_min, temp_max;
      std::tie(temp_min, temp_max) =
          filter_minmax_raw_array(filter_mapped_float, size, pbp);
      std::cout << "       minmax: " << temp_min << ", " << temp_max << '\n';
    }
#endif

    pbp[max] = false;

    changed_indices.push_back(max);

    if (!vulkan_get_filter(device, phys_atom_size, command_buffer, command_pool,
                           queue, pbp_buf, pipeline, pipeline_layout,
                           descriptor_set, filter_out_buf, size, pbp,
                           reversed_pbp, global_size, pbp_mapped_int,
                           staging_pbp_buffer, staging_pbp_buffer_mem,
                           staging_filter_buffer_mem, staging_filter_buffer,
                           &changed_indices)) {
      std::cerr << "Vulkan: Failed to execute do_filter\n";
      break;
    }

    // get second buffer's min
    int second_min;
    vulkan_minmax_opt = vulkan_minmax(
        device, phys_device, command_buffer, command_pool, queue,
        minmax_pipeline, minmax_pipeline_layout, minmax_descriptor_set,
        max_in_buf, min_in_buf, max_out_buf, min_out_buf, state_buf, size,
        filter_mapped_float, pbp);
    if (!vulkan_minmax_opt.has_value()) {
      std::cerr << "Vulkan: vulkan_minmax returned nullopt!\n";
      return {};
    }
    std::tie(second_min, std::ignore) = vulkan_minmax_opt.value();

    if (second_min == max) {
      pbp[max] = true;
      changed_indices.push_back(max);
      break;
    } else {
      pbp[second_min] = true;
      changed_indices.push_back(second_min);
    }

#ifndef NDEBUG
    if (iterations % 100 == 0) {
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
    }
#endif
  }

  if (!vulkan_get_filter(
          device, phys_atom_size, command_buffer, command_pool, queue, pbp_buf,
          pipeline, pipeline_layout, descriptor_set, filter_out_buf, size, pbp,
          reversed_pbp, global_size, pbp_mapped_int, staging_pbp_buffer,
          staging_pbp_buffer_mem, staging_filter_buffer_mem,
          staging_filter_buffer, &changed_indices)) {
    std::cerr << "Vulkan: Failed to execute do_filter (at end)\n";
  } else {
#ifndef NDEBUG
    internal::write_filter(vulkan_buf_to_vec(filter_mapped_float, size), width,
                           "filter_out_final.pgm");
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
  std::vector<unsigned int> dither_array(size, 0);
  int min, max;
  {
    std::vector<bool> pbp_copy(pbp);
    std::cout << "Ranking minority pixels...\n";
    for (unsigned int i = pixel_count; i-- > 0;) {
#ifndef NDEBUG
      std::cout << i << ' ';
#endif
      vulkan_get_filter(device, phys_atom_size, command_buffer, command_pool,
                        queue, pbp_buf, pipeline, pipeline_layout,
                        descriptor_set, filter_out_buf, size, pbp, reversed_pbp,
                        global_size, pbp_mapped_int, staging_pbp_buffer,
                        staging_pbp_buffer_mem, staging_filter_buffer_mem,
                        staging_filter_buffer, &changed_indices);
      auto vulkan_minmax_opt = vulkan_minmax(
          device, phys_device, command_buffer, command_pool, queue,
          minmax_pipeline, minmax_pipeline_layout, minmax_descriptor_set,
          max_in_buf, min_in_buf, max_out_buf, min_out_buf, state_buf, size,
          filter_mapped_float, pbp);
      if (!vulkan_minmax_opt.has_value()) {
        std::cerr << "Vulkan: vulkan_minmax returned nullopt!\n";
        return {};
      }
      std::tie(std::ignore, max) = vulkan_minmax_opt.value();
      pbp.at(max) = false;
      dither_array.at(max) = i;
      changed_indices.push_back(max);
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
  for (unsigned int i = pixel_count; i < (unsigned int)((size + 1) / 2); ++i) {
#ifndef NDEBUG
    std::cout << i << ' ';
#endif
    vulkan_get_filter(device, phys_atom_size, command_buffer, command_pool,
                      queue, pbp_buf, pipeline, pipeline_layout, descriptor_set,
                      filter_out_buf, size, pbp, reversed_pbp, global_size,
                      pbp_mapped_int, staging_pbp_buffer,
                      staging_pbp_buffer_mem, staging_filter_buffer_mem,
                      staging_filter_buffer, &changed_indices);
    auto vulkan_minmax_opt = vulkan_minmax(
        device, phys_device, command_buffer, command_pool, queue,
        minmax_pipeline, minmax_pipeline_layout, minmax_descriptor_set,
        max_in_buf, min_in_buf, max_out_buf, min_out_buf, state_buf, size,
        filter_mapped_float, pbp);
    if (!vulkan_minmax_opt.has_value()) {
      std::cerr << "Vulkan: vulkan_minmax returned nullopt!\n";
      return {};
    }
    std::tie(min, std::ignore) = vulkan_minmax_opt.value();
    pbp.at(min) = true;
    dither_array.at(min) = i;
    changed_indices.push_back(min);
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
    vulkan_get_filter(device, phys_atom_size, command_buffer, command_pool,
                      queue, pbp_buf, pipeline, pipeline_layout, descriptor_set,
                      filter_out_buf, size, pbp, reversed_pbp, global_size,
                      pbp_mapped_int, staging_pbp_buffer,
                      staging_pbp_buffer_mem, staging_filter_buffer_mem,
                      staging_filter_buffer, &changed_indices);
    internal::write_filter(vulkan_buf_to_vec(filter_mapped_float, size), width,
                           "filter_mid.pgm");
    image::Bl pbp_image = toBl(pbp, width);
    pbp_image.writeToFile(image::file_type::PNG, true, "debug_pbp_mid.png");
  }
#endif
  std::cout << "\nRanking last half of pixels...\n";
  reversed_pbp = true;
  bool first_reversed_run = true;
  for (unsigned int i = (size + 1) / 2; i < (unsigned int)size; ++i) {
#ifndef NDEBUG
    std::cout << i << ' ';
#endif
    if (first_reversed_run) {
      changed_indices.clear();
      first_reversed_run = false;
    }
    vulkan_get_filter(device, phys_atom_size, command_buffer, command_pool,
                      queue, pbp_buf, pipeline, pipeline_layout, descriptor_set,
                      filter_out_buf, size, pbp, reversed_pbp, global_size,
                      pbp_mapped_int, staging_pbp_buffer,
                      staging_pbp_buffer_mem, staging_filter_buffer_mem,
                      staging_filter_buffer, &changed_indices);
    auto vulkan_minmax_opt = vulkan_minmax(
        device, phys_device, command_buffer, command_pool, queue,
        minmax_pipeline, minmax_pipeline_layout, minmax_descriptor_set,
        max_in_buf, min_in_buf, max_out_buf, min_out_buf, state_buf, size,
        filter_mapped_float, pbp);
    if (!vulkan_minmax_opt.has_value()) {
      std::cerr << "Vulkan: vulkan_minmax returned nullopt!\n";
      return {};
    }
    std::tie(std::ignore, max) = vulkan_minmax_opt.value();
    pbp.at(max) = true;
    dither_array.at(max) = i;
    changed_indices.push_back(max);
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
    vulkan_get_filter(device, phys_atom_size, command_buffer, command_pool,
                      queue, pbp_buf, pipeline, pipeline_layout, descriptor_set,
                      filter_out_buf, size, pbp, reversed_pbp, global_size,
                      pbp_mapped_int, staging_pbp_buffer,
                      staging_pbp_buffer_mem, staging_filter_buffer_mem,
                      staging_filter_buffer, nullptr);
    internal::write_filter(vulkan_buf_to_vec(filter_mapped_float, size), width,
                           "filter_after.pgm");
    image::Bl pbp_image = toBl(pbp, width);
    pbp_image.writeToFile(image::file_type::PNG, true, "debug_pbp_after.png");
  }
#endif

  return dither_array;
}

std::vector<float> dither::internal::vulkan_buf_to_vec(float *mapped,
                                                       unsigned int size) {
  std::vector<float> v(size);

  std::memcpy(v.data(), mapped, size * sizeof(float));

  return v;
}

std::optional<std::pair<int, int>> dither::internal::vulkan_minmax(
    VkDevice device, VkPhysicalDevice phys_dev, VkCommandBuffer command_buffer,
    VkCommandPool command_pool, VkQueue queue, VkPipeline minmax_pipeline,
    VkPipelineLayout minmax_pipeline_layout,
    VkDescriptorSet minmax_descriptor_set, VkBuffer max_in_buf,
    VkBuffer min_in_buf, VkBuffer max_out_buf, VkBuffer min_out_buf,
    VkBuffer state_buf, const int size, const float *const filter_mapped,
    std::vector<bool> pbp) {
  // ensure minority pixel is "true"
  unsigned int count = 0;
  for (bool value : pbp) {
    if (value) {
      ++count;
    }
  }
  if (count * 2 >= pbp.size()) {
    // std::cout << "MINMAX flip\n"; // DEBUG
    for (unsigned int i = 0; i < pbp.size(); ++i) {
      pbp[i] = !pbp[i];
    }
  }

  std::vector<FloatAndIndex> fai(size);
  for (int i = 0; i < size; ++i) {
    fai[i].value = filter_mapped[i];
    fai[i].pbp = pbp[i] ? 1 : 0;
    fai[i].idx = i;
  }

  VkBuffer staging_buf;
  VkDeviceMemory staging_buf_mem;
  utility::Cleanup cleanup_staging_buf{};
  utility::Cleanup cleanup_staging_buf_mem{};
  void *staging_mapped;
  utility::Cleanup cleanup_staging_buf_mem_mapped{};
  VkMappedMemoryRange range{};
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(phys_dev, &props);
  {
    vulkan_create_buffer(
        device, phys_dev, size * sizeof(FloatAndIndex),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        staging_buf, staging_buf_mem);
    cleanup_staging_buf = utility::Cleanup(
        [device](void *ptr) {
          vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
        },
        &staging_buf);
    cleanup_staging_buf_mem = utility::Cleanup(
        [device](void *ptr) {
          vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
        },
        &staging_buf_mem);

    vkMapMemory(device, staging_buf_mem, 0, size * sizeof(FloatAndIndex), 0,
                &staging_mapped);
    cleanup_staging_buf_mem_mapped = utility::Cleanup(
        [device](void *ptr) {
          vkUnmapMemory(device, *((VkDeviceMemory *)ptr));
        },
        &staging_buf_mem);
    std::memcpy(staging_mapped, fai.data(), size * sizeof(FloatAndIndex));
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = staging_buf_mem;
    range.size = VK_WHOLE_SIZE;
    range.offset = 0;
    range.pNext = nullptr;

    vkFlushMappedMemoryRanges(device, 1, &range);

    vulkan_copy_buffer(device, command_pool, queue, staging_buf, max_in_buf,
                       size * sizeof(FloatAndIndex));
    vulkan_copy_buffer(device, command_pool, queue, staging_buf, min_in_buf,
                       size * sizeof(FloatAndIndex));

    fai[0].idx = size;
    std::memcpy(staging_mapped, &fai[0].idx, sizeof(int));

    if (sizeof(int) < props.limits.nonCoherentAtomSize) {
      range.size = props.limits.nonCoherentAtomSize;
    } else if (sizeof(int) > props.limits.nonCoherentAtomSize) {
      range.size = ((int)std::ceil((float)sizeof(int) /
                                   (float)props.limits.nonCoherentAtomSize)) *
                   props.limits.nonCoherentAtomSize;
    } else {
      range.size = props.limits.nonCoherentAtomSize;
    }
    vkFlushMappedMemoryRanges(device, 1, &range);

    vulkan_copy_buffer(device, command_pool, queue, staging_buf, state_buf,
                       sizeof(int));
  }

  int current_size = size;
  int next_size;
  while (current_size > 1) {
    next_size = (current_size + 1) / 2;
    vkResetCommandBuffer(command_buffer, 0);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
      std::clog << "vulkan_minmax ERROR: Failed to begin record compute "
                   "command buffer!\n";
      return std::nullopt;
    }

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      minmax_pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            minmax_pipeline_layout, 0, 1,
                            &minmax_descriptor_set, 0, nullptr);
    vkCmdDispatch(command_buffer, std::ceil((float)next_size / 256.0F), 1, 1);
    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
      std::clog
          << "vulkan_minmax ERROR: Failed to record compute command buffer!\n";
      return std::nullopt;
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    submit_info.signalSemaphoreCount = 0;
    submit_info.pSignalSemaphores = nullptr;

    if (vkQueueSubmit(queue, 1, &submit_info, nullptr) != VK_SUCCESS) {
      std::clog
          << "vulkan_minmax ERROR: Failed to submit compute command buffer!\n";
      return std::nullopt;
    }

    if (vkDeviceWaitIdle(device) != VK_SUCCESS) {
      std::clog << "vulkan_minmax ERROR: Failed to vkDeviceWaitIdle!\n";
      return std::nullopt;
    }

    if (next_size > 1) {
      vulkan_copy_buffer(device, command_pool, queue, max_out_buf, max_in_buf,
                         next_size * sizeof(FloatAndIndex));
      vulkan_copy_buffer(device, command_pool, queue, min_out_buf, min_in_buf,
                         next_size * sizeof(FloatAndIndex));

      fai[0].idx = next_size;
      std::memcpy(staging_mapped, &fai[0].idx, sizeof(int));
      vkFlushMappedMemoryRanges(device, 1, &range);
      vulkan_copy_buffer(device, command_pool, queue, staging_buf, state_buf,
                         sizeof(int));
    }

    current_size = next_size;
  }

  vulkan_copy_buffer(device, command_pool, queue, min_out_buf, staging_buf,
                     sizeof(FloatAndIndex), 0, 0);
  vulkan_copy_buffer(device, command_pool, queue, max_out_buf, staging_buf,
                     sizeof(FloatAndIndex), 0, sizeof(FloatAndIndex));

  if (sizeof(FloatAndIndex) * 2 < props.limits.nonCoherentAtomSize) {
    range.size = props.limits.nonCoherentAtomSize;
  } else if (sizeof(FloatAndIndex) * 2 > props.limits.nonCoherentAtomSize) {
    range.size = ((int)std::ceil((float)sizeof(FloatAndIndex) * 2.0F /
                                 (float)props.limits.nonCoherentAtomSize)) *
                 props.limits.nonCoherentAtomSize;
  } else {
    range.size = props.limits.nonCoherentAtomSize;
  }
  vkInvalidateMappedMemoryRanges(device, 1, &range);

  return std::make_pair(((FloatAndIndex *)staging_mapped)->idx,
                        (((FloatAndIndex *)staging_mapped) + 1)->idx);
}

#endif  // DITHERING_VULKAN_ENABLED == 1

#include "image.hpp"

image::Bl dither::blue_noise(int width, int height, int threads,
                             bool use_opencl, bool use_vulkan) {
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

    VkInstance instance;
    utility::Cleanup cleanup_vk_instance{};
    VkDebugUtilsMessengerEXT debug_messenger;
    utility::Cleanup cleanup_debug_messenger{};
    {
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
            info->sType =
                VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
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
      if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
        std::clog << "WARNING: Failed to create Vulkan instance!\n";
        goto ENDOF_VULKAN;
      }
      cleanup_vk_instance = utility::Cleanup(
          [](void *ptr) { vkDestroyInstance(*((VkInstance *)ptr), nullptr); },
          &instance);

#if VULKAN_VALIDATION == 1
      populate_debug_info(&debug_create_info);

      auto create_debug_utils_messenger_func =
          (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
              instance, "vkCreateDebugUtilsMessengerEXT");
      if (create_debug_utils_messenger_func == nullptr ||
          create_debug_utils_messenger_func(instance, &debug_create_info,
                                            nullptr,
                                            &debug_messenger) != VK_SUCCESS) {
        std::clog << "WARNING: Failed to set up Vulkan debug messenger!\n";
        goto ENDOF_VULKAN;
      }
      cleanup_debug_messenger = utility::Cleanup(
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
    }

    VkPhysicalDevice phys_device;
    {
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
        auto indices = internal::vulkan_find_queue_families(device);

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

      if (gpu_dev_discrete.has_value()) {
        std::clog << "NOTICE: Found discrete GPU supporting Vulkan compute.\n";
        phys_device = gpu_dev_discrete.value();
      } else if (gpu_dev_integrated.has_value()) {
        std::clog
            << "NOTICE: Found integrated GPU supporting Vulkan compute.\n";
        phys_device = gpu_dev_integrated.value();
      } else {
        std::clog << "WARNING: No suitable GPUs found!\n";
        goto ENDOF_VULKAN;
      }
    }

    VkDevice device;
    utility::Cleanup device_cleanup{};
    {
      auto indices = internal::vulkan_find_queue_families(phys_device);
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

      if (vkCreateDevice(phys_device, &dev_create_info, nullptr, &device) !=
          VK_SUCCESS) {
        std::clog << "WARNING: Failed to create VkDevice!\n";
        goto ENDOF_VULKAN;
      }
      device_cleanup = utility::Cleanup(
          [](void *ptr) { vkDestroyDevice(*((VkDevice *)ptr), nullptr); },
          &device);
    }

    VkQueue compute_queue;
    vkGetDeviceQueue(
        device,
        internal::vulkan_find_queue_families(phys_device).computeFamily.value(),
        0, &compute_queue);

    VkDescriptorSetLayout compute_desc_set_layout;
    utility::Cleanup compute_desc_set_layout_cleanup{};
    {
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

      if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr,
                                      &compute_desc_set_layout) != VK_SUCCESS) {
        std::clog
            << "WARNING: Failed to create compute descriptor set layout!\n";
        goto ENDOF_VULKAN;
      }
      compute_desc_set_layout_cleanup = utility::Cleanup(
          [device](void *ptr) {
            vkDestroyDescriptorSetLayout(
                device, *((VkDescriptorSetLayout *)ptr), nullptr);
          },
          &compute_desc_set_layout);
    }

    VkDescriptorSetLayout minmax_compute_desc_set_layout;
    utility::Cleanup cleanup_minmax_compute_desc_set_layout{};
    {
      std::array<VkDescriptorSetLayoutBinding, 5> compute_layout_bindings{};
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

      compute_layout_bindings[4].binding = 4;
      compute_layout_bindings[4].descriptorCount = 1;
      compute_layout_bindings[4].descriptorType =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      compute_layout_bindings[4].pImmutableSamplers = nullptr;
      compute_layout_bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

      VkDescriptorSetLayoutCreateInfo layout_info{};
      layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      layout_info.bindingCount = compute_layout_bindings.size();
      layout_info.pBindings = compute_layout_bindings.data();

      if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr,
                                      &minmax_compute_desc_set_layout) !=
          VK_SUCCESS) {
        std::clog << "WARNING: Failed to create compute descriptor set layout "
                     "(minmax)!\n";
        goto ENDOF_VULKAN;
      }
      cleanup_minmax_compute_desc_set_layout = utility::Cleanup(
          [device](void *ptr) {
            vkDestroyDescriptorSetLayout(
                device, *((VkDescriptorSetLayout *)ptr), nullptr);
          },
          &minmax_compute_desc_set_layout);
    }

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

      std::array<const char *, 3> minmax_filenames{
          "blue_noise_minmax.glsl", "src/blue_noise_minmax.glsl",
          "../src/blue_noise_minmax.glsl"};
      success = false;
      for (const auto filename : minmax_filenames) {
        std::ifstream ifs(filename);
        if (ifs.good()) {
          ifs.close();
          std::string command(
              "glslc -fshader-stage=compute -o compute_minmax.spv ");
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
        std::clog << "WARNING: Could not find blue_noise_minmax.glsl!\n";
        goto ENDOF_VULKAN;
      }
    }

    // create compute pipeline.
    VkPipelineLayout compute_pipeline_layout;
    VkPipeline compute_pipeline;
    utility::Cleanup cleanup_pipeline_layout{};
    utility::Cleanup cleanup_pipeline{};
    {
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

    VkPipelineLayout minmax_compute_pipeline_layout;
    VkPipeline minmax_compute_pipeline;
    utility::Cleanup cleanup_minmax_pipeline_layout{};
    utility::Cleanup cleanup_minmax_pipeline{};
    {
      // Load shader.
      std::vector<char> shader;
      {
        std::ifstream ifs("compute_minmax.spv");
        if (!ifs.good()) {
          std::clog << "WARNING: Failed to find compute_minmax.spv!\n";
          goto ENDOF_VULKAN;
        }
        ifs.seekg(0, std::ios_base::end);
        auto size = ifs.tellg();
        shader.resize(size);

        ifs.seekg(0);
        ifs.read(shader.data(), size);
        ifs.close();
      }

      VkShaderModuleCreateInfo shader_module_create_info{};
      shader_module_create_info.sType =
          VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      shader_module_create_info.codeSize = shader.size();
      shader_module_create_info.pCode =
          reinterpret_cast<const uint32_t *>(shader.data());

      VkShaderModule compute_shader_module;
      if (vkCreateShaderModule(device, &shader_module_create_info, nullptr,
                               &compute_shader_module) != VK_SUCCESS) {
        std::clog << "WARNING: Failed to create shader module (minmax)!\n";
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
      pipeline_layout_info.pSetLayouts = &minmax_compute_desc_set_layout;

      if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr,
                                 &minmax_compute_pipeline_layout) !=
          VK_SUCCESS) {
        std::clog
            << "WARNING: Failed to create compute pipeline layout (minmax)!\n";
        goto ENDOF_VULKAN;
      }
      cleanup_minmax_pipeline_layout = utility::Cleanup(
          [device](void *ptr) {
            vkDestroyPipelineLayout(device, *((VkPipelineLayout *)ptr),
                                    nullptr);
          },
          &minmax_compute_pipeline_layout);

      VkComputePipelineCreateInfo pipeline_info{};
      pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      pipeline_info.layout = minmax_compute_pipeline_layout;
      pipeline_info.stage = compute_shader_stage_info;

      if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   nullptr,
                                   &minmax_compute_pipeline) != VK_SUCCESS) {
        std::clog << "WARNING: Failed to create compute pipeline (minmax)!\n";
        goto ENDOF_VULKAN;
      }
      cleanup_minmax_pipeline = utility::Cleanup(
          [device](void *ptr) {
            vkDestroyPipeline(device, *((VkPipeline *)ptr), nullptr);
          },
          &minmax_compute_pipeline);
    }

    VkCommandPool command_pool;
    {
      VkCommandPoolCreateInfo pool_info{};
      pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      pool_info.queueFamilyIndex =
          internal::vulkan_find_queue_families(phys_device)
              .computeFamily.value();

      if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) !=
          VK_SUCCESS) {
        std::clog << "WARNING: Failed to create vulkan command pool!\n";
        goto ENDOF_VULKAN;
      }
    }
    utility::Cleanup cleanup_command_pool(
        [device](void *ptr) {
          vkDestroyCommandPool(device, *((VkCommandPool *)ptr), nullptr);
        },
        &command_pool);

    int filter_size = (width + height) / 2;
    std::vector<float> precomputed = internal::precompute_gaussian(filter_size);
    VkDeviceSize precomputed_size = sizeof(float) * precomputed.size();
    VkDeviceSize filter_out_size = sizeof(float) * width * height;
    VkDeviceSize pbp_size = sizeof(int) * width * height;
    VkDeviceSize other_size = sizeof(int) * 3;

    VkBuffer precomputed_buf;
    VkDeviceMemory precomputed_buf_mem;
    utility::Cleanup cleanup_precomputed_buf{};
    utility::Cleanup cleanup_precomputed_buf_mem{};
    {
      VkBuffer staging_buffer;
      VkDeviceMemory staging_buffer_mem;

      if (!internal::vulkan_create_buffer(
              device, phys_device, precomputed_size,
              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              staging_buffer, staging_buffer_mem)) {
        std::clog << "WARNING: Failed to create staging buffer!\n";
        goto ENDOF_VULKAN;
      }
      utility::Cleanup cleanup_staging_buf(
          [device](void *ptr) {
            vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
          },
          &staging_buffer);
      utility::Cleanup cleanup_staging_buf_mem(
          [device](void *ptr) {
            vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
          },
          &staging_buffer_mem);

      void *data_ptr;
      vkMapMemory(device, staging_buffer_mem, 0, precomputed_size, 0,
                  &data_ptr);
      std::memcpy(data_ptr, precomputed.data(), precomputed_size);
      vkUnmapMemory(device, staging_buffer_mem);

      if (!internal::vulkan_create_buffer(device, phys_device, precomputed_size,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                          precomputed_buf,
                                          precomputed_buf_mem)) {
        std::clog << "WARNING: Failed to create precomputed buffer!\n";
        goto ENDOF_VULKAN;
      }
      cleanup_precomputed_buf = utility::Cleanup(
          [device](void *ptr) {
            vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
          },
          &precomputed_buf);
      cleanup_precomputed_buf_mem = utility::Cleanup(
          [device](void *ptr) {
            vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
          },
          &precomputed_buf_mem);

      internal::vulkan_copy_buffer(device, command_pool, compute_queue,
                                   staging_buffer, precomputed_buf,
                                   precomputed_size);
    }

    VkBuffer filter_out_buf;
    VkDeviceMemory filter_out_buf_mem;
    if (!internal::vulkan_create_buffer(device, phys_device, filter_out_size,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                        filter_out_buf, filter_out_buf_mem)) {
      std::clog << "WARNING: Failed to create filter_out buffer!\n";
      goto ENDOF_VULKAN;
    }
    utility::Cleanup cleanup_filter_out_buf(
        [device](void *ptr) {
          vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
        },
        &filter_out_buf);
    utility::Cleanup cleanup_filter_out_buf_mem(
        [device](void *ptr) {
          vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
        },
        &filter_out_buf_mem);

    VkBuffer pbp_buf;
    VkDeviceMemory pbp_buf_mem;
    if (!internal::vulkan_create_buffer(device, phys_device, pbp_size,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                        pbp_buf, pbp_buf_mem)) {
      std::clog << "WARNING: Failed to create pbp buffer!\n";
      goto ENDOF_VULKAN;
    }
    utility::Cleanup cleanup_pbp_buf(
        [device](void *ptr) {
          vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
        },
        &pbp_buf);
    utility::Cleanup cleanup_pbp_buf_mem(
        [device](void *ptr) {
          vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
        },
        &pbp_buf_mem);

    VkBuffer other_buf;
    VkDeviceMemory other_buf_mem;
    utility::Cleanup cleanup_other_buf{};
    utility::Cleanup cleanup_other_buf_mem{};
    {
      VkBuffer staging_buffer;
      VkDeviceMemory staging_buffer_mem;

      if (!internal::vulkan_create_buffer(
              device, phys_device, other_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              staging_buffer, staging_buffer_mem)) {
        std::clog << "WARNING: Failed to create staging buffer!\n";
        goto ENDOF_VULKAN;
      }
      utility::Cleanup cleanup_staging_buf(
          [device](void *ptr) {
            vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
          },
          &staging_buffer);
      utility::Cleanup cleanup_staging_buf_mem(
          [device](void *ptr) {
            vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
          },
          &staging_buffer_mem);

      void *data_ptr;
      vkMapMemory(device, staging_buffer_mem, 0, other_size, 0, &data_ptr);
      std::memcpy(data_ptr, &width, sizeof(int));
      std::memcpy(((char *)data_ptr) + sizeof(int), &height, sizeof(int));
      if (filter_size % 2 == 0) {
        int filter_size_odd = filter_size + 1;
        std::memcpy(((char *)data_ptr) + sizeof(int) * 2, &filter_size_odd,
                    sizeof(int));
      } else {
        std::memcpy(((char *)data_ptr) + sizeof(int) * 2, &filter_size,
                    sizeof(int));
      }
      vkUnmapMemory(device, staging_buffer_mem);

      if (!internal::vulkan_create_buffer(device, phys_device, other_size,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                          other_buf, other_buf_mem)) {
        std::clog << "WARNING: Failed to create other buffer!\n";
        goto ENDOF_VULKAN;
      }
      cleanup_other_buf = utility::Cleanup(
          [device](void *ptr) {
            vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
          },
          &other_buf);
      cleanup_other_buf_mem = utility::Cleanup(
          [device](void *ptr) {
            vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
          },
          &other_buf_mem);

      internal::vulkan_copy_buffer(device, command_pool, compute_queue,
                                   staging_buffer, other_buf, other_size);
    }

    VkBuffer max_in_buf;
    VkBuffer min_in_buf;
    VkBuffer min_out_buf;
    VkBuffer max_out_buf;
    VkBuffer state_buf;
    VkDeviceMemory max_in_buf_mem;
    VkDeviceMemory min_in_buf_mem;
    VkDeviceMemory min_out_buf_mem;
    VkDeviceMemory max_out_buf_mem;
    VkDeviceMemory state_buf_mem;
    if (!internal::vulkan_create_buffer(
            device, phys_device,
            width * height * sizeof(dither::internal::FloatAndIndex),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, max_in_buf, max_in_buf_mem)) {
      std::clog << "WARNING: Failed to create max_in buffer (minmax)!\n";
      goto ENDOF_VULKAN;
    }
    utility::Cleanup cleanup_max_in_buf(
        [device](void *ptr) {
          vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
        },
        &max_in_buf);
    utility::Cleanup cleanup_max_in_buf_mem(
        [device](void *ptr) {
          vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
        },
        &max_in_buf_mem);

    if (!internal::vulkan_create_buffer(
            device, phys_device,
            width * height * sizeof(dither::internal::FloatAndIndex),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, min_in_buf, min_in_buf_mem)) {
      std::clog << "WARNING: Failed to create min_in buffer (minmax)!\n";
      goto ENDOF_VULKAN;
    }
    utility::Cleanup cleanup_min_in_buf(
        [device](void *ptr) {
          vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
        },
        &min_in_buf);
    utility::Cleanup cleanup_min_in_buf_mem(
        [device](void *ptr) {
          vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
        },
        &min_in_buf_mem);

    if (!internal::vulkan_create_buffer(
            device, phys_device,
            ((width * height + 1) / 2) *
                sizeof(dither::internal::FloatAndIndex),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, min_out_buf,
            min_out_buf_mem)) {
      std::clog << "WARNING: Failed to create min_out buffer (minmax)!\n";
      goto ENDOF_VULKAN;
    }
    utility::Cleanup cleanup_min_out_buf(
        [device](void *ptr) {
          vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
        },
        &min_out_buf);
    utility::Cleanup cleanup_min_out_buf_mem(
        [device](void *ptr) {
          vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
        },
        &min_out_buf_mem);

    if (!internal::vulkan_create_buffer(
            device, phys_device,
            ((width * height + 1) / 2) *
                sizeof(dither::internal::FloatAndIndex),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, max_out_buf,
            max_out_buf_mem)) {
      std::clog << "WARNING: Failed to create max_out buffer (minmax)!\n";
      goto ENDOF_VULKAN;
    }
    utility::Cleanup cleanup_max_out_buf(
        [device](void *ptr) {
          vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
        },
        &max_out_buf);
    utility::Cleanup cleanup_max_out_buf_mem(
        [device](void *ptr) {
          vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
        },
        &max_out_buf_mem);

    if (!internal::vulkan_create_buffer(device, phys_device, sizeof(int),
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                        state_buf, state_buf_mem)) {
      std::clog << "WARNING: Failed to create state buffer (minmax)!\n";
      goto ENDOF_VULKAN;
    }
    utility::Cleanup cleanup_state_buf(
        [device](void *ptr) {
          vkDestroyBuffer(device, *((VkBuffer *)ptr), nullptr);
        },
        &state_buf);
    utility::Cleanup cleanup_state_buf_mem(
        [device](void *ptr) {
          vkFreeMemory(device, *((VkDeviceMemory *)ptr), nullptr);
        },
        &state_buf_mem);

    VkDescriptorPool descriptor_pool;
    utility::Cleanup cleanup_descriptor_pool{};
    {
      VkDescriptorPoolSize pool_size{};
      pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      pool_size.descriptorCount = 4;

      VkDescriptorPoolCreateInfo pool_info{};
      pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      pool_info.poolSizeCount = 1;
      pool_info.pPoolSizes = &pool_size;
      pool_info.maxSets = 1;

      if (vkCreateDescriptorPool(device, &pool_info, nullptr,
                                 &descriptor_pool) != VK_SUCCESS) {
        std::clog << "WARNING: Failed to create descriptor pool!\n";
        goto ENDOF_VULKAN;
      }

      cleanup_descriptor_pool = utility::Cleanup(
          [device](void *ptr) {
            vkDestroyDescriptorPool(device, *((VkDescriptorPool *)ptr),
                                    nullptr);
          },
          &descriptor_pool);
    }

    VkDescriptorPool minmax_descriptor_pool;
    utility::Cleanup cleanup_minmax_descriptor_pool{};
    {
      VkDescriptorPoolSize pool_size{};
      pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      pool_size.descriptorCount = 5;

      VkDescriptorPoolCreateInfo pool_info{};
      pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      pool_info.poolSizeCount = 1;
      pool_info.pPoolSizes = &pool_size;
      pool_info.maxSets = 1;

      if (vkCreateDescriptorPool(device, &pool_info, nullptr,
                                 &minmax_descriptor_pool) != VK_SUCCESS) {
        std::clog << "WARNING: Failed to create descriptor pool (minmax)!\n";
        goto ENDOF_VULKAN;
      }

      cleanup_minmax_descriptor_pool = utility::Cleanup(
          [device](void *ptr) {
            vkDestroyDescriptorPool(device, *((VkDescriptorPool *)ptr),
                                    nullptr);
          },
          &minmax_descriptor_pool);
    }

    VkDescriptorSet compute_descriptor_set;
    {
      VkDescriptorSetAllocateInfo alloc_info{};
      alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      alloc_info.descriptorPool = descriptor_pool;
      alloc_info.descriptorSetCount = 1;
      alloc_info.pSetLayouts = &compute_desc_set_layout;

      if (vkAllocateDescriptorSets(device, &alloc_info,
                                   &compute_descriptor_set) != VK_SUCCESS) {
        std::clog << "WARNING: Failed to allocate descriptor set!\n";
        goto ENDOF_VULKAN;
      }

      std::array<VkWriteDescriptorSet, 4> descriptor_writes{};

      descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptor_writes[0].dstSet = compute_descriptor_set;
      descriptor_writes[0].dstBinding = 0;
      descriptor_writes[0].dstArrayElement = 0;
      descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_writes[0].descriptorCount = 1;
      VkDescriptorBufferInfo precomputed_info{};
      precomputed_info.buffer = precomputed_buf;
      precomputed_info.offset = 0;
      precomputed_info.range = VK_WHOLE_SIZE;
      descriptor_writes[0].pBufferInfo = &precomputed_info;

      descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptor_writes[1].dstSet = compute_descriptor_set;
      descriptor_writes[1].dstBinding = 1;
      descriptor_writes[1].dstArrayElement = 0;
      descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_writes[1].descriptorCount = 1;
      VkDescriptorBufferInfo filter_out_info{};
      filter_out_info.buffer = filter_out_buf;
      filter_out_info.offset = 0;
      filter_out_info.range = VK_WHOLE_SIZE;
      descriptor_writes[1].pBufferInfo = &filter_out_info;

      descriptor_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptor_writes[2].dstSet = compute_descriptor_set;
      descriptor_writes[2].dstBinding = 2;
      descriptor_writes[2].dstArrayElement = 0;
      descriptor_writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_writes[2].descriptorCount = 1;
      VkDescriptorBufferInfo pbp_info{};
      pbp_info.buffer = pbp_buf;
      pbp_info.offset = 0;
      pbp_info.range = VK_WHOLE_SIZE;
      descriptor_writes[2].pBufferInfo = &pbp_info;

      descriptor_writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptor_writes[3].dstSet = compute_descriptor_set;
      descriptor_writes[3].dstBinding = 3;
      descriptor_writes[3].dstArrayElement = 0;
      descriptor_writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_writes[3].descriptorCount = 1;
      VkDescriptorBufferInfo other_info{};
      other_info.buffer = other_buf;
      other_info.offset = 0;
      other_info.range = VK_WHOLE_SIZE;
      descriptor_writes[3].pBufferInfo = &other_info;

      vkUpdateDescriptorSets(device, descriptor_writes.size(),
                             descriptor_writes.data(), 0, nullptr);
    }

    VkDescriptorSet minmax_compute_descriptor_set;
    {
      VkDescriptorSetAllocateInfo alloc_info{};
      alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      alloc_info.descriptorPool = minmax_descriptor_pool;
      alloc_info.descriptorSetCount = 1;
      alloc_info.pSetLayouts = &minmax_compute_desc_set_layout;

      if (vkAllocateDescriptorSets(device, &alloc_info,
                                   &minmax_compute_descriptor_set) !=
          VK_SUCCESS) {
        std::clog << "WARNING: Failed to allocate descriptor set (minmax)!\n";
        goto ENDOF_VULKAN;
      }

      std::array<VkWriteDescriptorSet, 5> descriptor_writes{};

      descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptor_writes[0].dstSet = minmax_compute_descriptor_set;
      descriptor_writes[0].dstBinding = 0;
      descriptor_writes[0].dstArrayElement = 0;
      descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_writes[0].descriptorCount = 1;
      VkDescriptorBufferInfo max_in_info{};
      max_in_info.buffer = max_in_buf;
      max_in_info.offset = 0;
      max_in_info.range = VK_WHOLE_SIZE;
      descriptor_writes[0].pBufferInfo = &max_in_info;

      descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptor_writes[1].dstSet = minmax_compute_descriptor_set;
      descriptor_writes[1].dstBinding = 1;
      descriptor_writes[1].dstArrayElement = 0;
      descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_writes[1].descriptorCount = 1;
      VkDescriptorBufferInfo min_in_info{};
      min_in_info.buffer = min_in_buf;
      min_in_info.offset = 0;
      min_in_info.range = VK_WHOLE_SIZE;
      descriptor_writes[1].pBufferInfo = &min_in_info;

      descriptor_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptor_writes[2].dstSet = minmax_compute_descriptor_set;
      descriptor_writes[2].dstBinding = 2;
      descriptor_writes[2].dstArrayElement = 0;
      descriptor_writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_writes[2].descriptorCount = 1;
      VkDescriptorBufferInfo max_out_info{};
      max_out_info.buffer = max_out_buf;
      max_out_info.offset = 0;
      max_out_info.range = VK_WHOLE_SIZE;
      descriptor_writes[2].pBufferInfo = &max_out_info;

      descriptor_writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptor_writes[3].dstSet = minmax_compute_descriptor_set;
      descriptor_writes[3].dstBinding = 3;
      descriptor_writes[3].dstArrayElement = 0;
      descriptor_writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_writes[3].descriptorCount = 1;
      VkDescriptorBufferInfo min_out_info{};
      min_out_info.buffer = min_out_buf;
      min_out_info.offset = 0;
      min_out_info.range = VK_WHOLE_SIZE;
      descriptor_writes[3].pBufferInfo = &min_out_info;

      descriptor_writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptor_writes[4].dstSet = minmax_compute_descriptor_set;
      descriptor_writes[4].dstBinding = 4;
      descriptor_writes[4].dstArrayElement = 0;
      descriptor_writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_writes[4].descriptorCount = 1;
      VkDescriptorBufferInfo state_info{};
      state_info.buffer = state_buf;
      state_info.offset = 0;
      state_info.range = VK_WHOLE_SIZE;
      descriptor_writes[4].pBufferInfo = &state_info;

      vkUpdateDescriptorSets(device, descriptor_writes.size(),
                             descriptor_writes.data(), 0, nullptr);
    }

    VkCommandBuffer command_buffer;
    {
      VkCommandBufferAllocateInfo alloc_info{};
      alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      alloc_info.commandPool = command_pool;
      alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      alloc_info.commandBufferCount = 1;

      if (vkAllocateCommandBuffers(device, &alloc_info, &command_buffer) !=
          VK_SUCCESS) {
        std::clog << "WARNING: Failed to allocate compute command buffer!\n";
        goto ENDOF_VULKAN;
      }
    }

    auto result = dither::internal::blue_noise_vulkan_impl(
        device, phys_device, command_buffer, command_pool, compute_queue,
        pbp_buf, compute_pipeline, compute_pipeline_layout,
        compute_descriptor_set, filter_out_buf, minmax_compute_pipeline,
        minmax_compute_pipeline_layout, minmax_compute_descriptor_set,
        max_in_buf, min_in_buf, max_out_buf, min_out_buf, state_buf, width,
        height);
    if (!result.empty()) {
      return internal::rangeToBl(result, width);
    }
    std::cout << "ERROR: Empty result\n";
    return {};
  }
ENDOF_VULKAN:
#else
  std::clog << "WARNING: Not compiled with Vulkan support!\n";
#endif  // DITHERING_VULKAN_ENABLED == 1

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
