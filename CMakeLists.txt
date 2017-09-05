cmake_minimum_required(VERSION 2.8.12)
project(_pqkmeans)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter")

add_subdirectory(ext/pybind)
file(GLOB SOURCES src/**/*.cpp)
pybind11_add_module(_pqkmeans ${SOURCES})
