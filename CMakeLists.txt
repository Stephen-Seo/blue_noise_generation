cmake_minimum_required(VERSION 3.9)
project(blueNoiseGen)

set(blueNoiseGen_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blue_noise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/arg_parse.cpp
)

add_compile_options(
    -Wall -Wextra -Wpedantic
    $<$<COMPILE_LANGUAGE:CXX>:-Weffc++>
    $<$<CONFIG:DEBUG>:-O0>
)

if(NOT DEFINED CMAKE_BUILD_TYPE OR NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
    message("Set build type to Debug by default")
endif()

find_package(Threads REQUIRED)
if(NOT DEFINED DISABLE_OPENCL OR NOT DISABLE_OPENCL)
    find_package(OpenCL)
    if(NOT DEFINED OpenCL_FOUND)
        set(DISABLE_OPENCL True)
        message(WARNING "OpenCL not found, OpenCL usage is disabled.")
    endif()
else()
    message(STATUS "Not checking for OpenCL")
endif()
find_package(PNG REQUIRED)

add_executable(blueNoiseGen ${blueNoiseGen_SOURCES})
target_compile_features(blueNoiseGen PUBLIC cxx_std_17)
target_compile_definitions(blueNoiseGen PRIVATE CL_TARGET_OPENCL_VERSION=300)
if(DEFINED DISABLE_OPENCL AND DISABLE_OPENCL)
    message(STATUS "OpenCL usage is disabled.")
    target_include_directories(blueNoiseGen PUBLIC
        Threads::Threads
        ${PNG_INCLUDE_DIRS})
    target_link_libraries(blueNoiseGen PUBLIC
        Threads::Threads
        ${PNG_LIBRARIES})
    target_compile_definitions(blueNoiseGen PRIVATE DITHERING_OPENCL_ENABLED=0)
else()
    message(STATUS "OpenCL usage is enabled.")
    target_include_directories(blueNoiseGen PUBLIC
        Threads::Threads
        ${OpenCL_INCLUDE_DIRS}
        ${PNG_INCLUDE_DIRS})
    target_link_libraries(blueNoiseGen PUBLIC
        Threads::Threads
        ${OpenCL_LIBRARIES}
        ${PNG_LIBRARIES})
    target_compile_definitions(blueNoiseGen PRIVATE DITHERING_OPENCL_ENABLED=1)
endif()
