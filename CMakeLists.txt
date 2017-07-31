cmake_minimum_required(VERSION 2.8.12)
project(_pqkmeans)

add_subdirectory(ext/pybind)
pybind11_add_module(_pqkmeans src/_pqkmeans.cpp src/encoder/encoder_sample.cpp)
