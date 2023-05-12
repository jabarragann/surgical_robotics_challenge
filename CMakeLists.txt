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

# Test yaml-cpp
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_INCLUDE_DIRS})

set(yaml_prefix "plugins/yaml_test")
add_executable(yaml_test ${yaml_prefix}/read_yaml.cpp)
target_link_libraries(yaml_test ${YAML_CPP_LIBRARIES})