cmake_minimum_required(VERSION 3.1)
project(volumetric_plugin)

set(CMAKE_CXX_STANDARD 11)

find_package(AMBF)
find_package(Boost COMPONENTS program_options filesystem)

include_directories(${AMBF_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

link_directories(${AMBF_LIBRARY_DIRS})

add_definitions(${AMBF_DEFINITIONS})

set(shaders_plugin_prefix "plugins/PreprocessingShadersConfig")

add_library(prepropressing_shader_config SHARED
    ${shaders_plugin_prefix}/shaders_config.cpp
    ${shaders_plugin_prefix}/shaders_config.h)

target_link_libraries(prepropressing_shader_config ${Boost_LIBRARIES} ${AMBF_LIBRARIES})
set_property(TARGET prepropressing_shader_config PROPERTY POSITION_INDEPENDENT_CODE TRUE)
