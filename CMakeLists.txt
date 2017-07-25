cmake_minimum_required(VERSION 3.6)
# please use the comaptible version of cmake according to the version of your boost library

set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
set(CMAKE_CXX_STANDARD 11)

find_package(Boost COMPONENTS python3 REQUIRED) # for python3
find_package(PythonLibs REQUIRED)
include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

ADD_LIBRARY(_pqkmeans SHARED src/_pqkmeans.cpp src/encoder/encoder_sample.cpp src/encoder/encoder_sample.h)
TARGET_LINK_LIBRARIES(_pqkmeans ${Boost_LIBRARIES} ${PYTHON_LIBRARY})
set_target_properties(_pqkmeans PROPERTIES PREFIX "")
