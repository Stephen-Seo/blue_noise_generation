cmake_minimum_required(VERSION 3.9)
project(blueNoiseGen)

set(blueNoiseGen_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blue_noise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/arg_parse.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

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
