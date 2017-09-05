cmake_minimum_required(VERSION 2.8.12)
project(_pqkmeans)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter")

add_subdirectory(ext/pybind)
pybind11_add_module(_pqkmeans src/_pqkmeans.cpp src/encoder/encoder_sample.cpp src/clustering/cpp_implemented_clustering_sample.cpp)
