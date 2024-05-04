include(config.cmake)

cmake_minimum_required(VERSION 3.10)

# Setting Build Configurations
project(VNN LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Adding Libraries / Source Codes
file(GLOB SOURCES "lib/*.cxx")
file(GLOB LAYERS_SOURCES "lib/layers/*.cxx")

# Created a shared library
add_library(vnn SHARED ${SOURCES} ${LAYERS_SOURCES})

# Setting Imports
set(IMPORTS ${EIGEN_DIR} "./headers")
include_directories(${IMPORTS})
target_include_directories(vnn PUBLIC ${IMPORTS})

install(TARGETS vnn DESTINATION lib)
install(DIRECTORY headers/ DESTINATION include)