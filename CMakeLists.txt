cmake_minimum_required(VERSION 3.10)

# Set the project name
project(Matrices)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)


include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/lib/include)
# Add the executable
add_executable(Matrices src/main.cpp)