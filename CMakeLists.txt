cmake_minimum_required(VERSION 2.8.12)
project(_pqkmeans)

find_package( OpenMP )
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")

add_subdirectory(ext/pybind)
file(GLOB SOURCES src/_pqkmeans.cpp src/**/*.cpp)
pybind11_add_module(_pqkmeans ${SOURCES})
