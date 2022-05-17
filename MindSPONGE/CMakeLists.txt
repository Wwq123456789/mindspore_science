cmake_minimum_required(VERSION 3.14.1)
project(MindSPONGE)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(FATAL_ERROR "MindSPONGE can not support Windows")
endif()

option(ENABLE_D "Enable d" OFF)
option(ENABLE_GPU "Enable gpu" OFF)

set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/mindsponge/libs)
message("CMAKE_SOURCE_DIR is ${CMAKE_SOURCE_DIR}")

ADD_SUBDIRECTORY(mindsponge/ccsrc/backend/)
