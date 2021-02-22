cmake_minimum_required(VERSION 2.8.12)
project(multiprofit)
# Set source directory
set(SOURCE_DIR "src")
# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})
set(SOURCES "")
# Generate Python module
find_package(pybind11 REQUIRED)
pybind11_add_module(_multiprofit ${SOURCES} "multiprofit/pybind11.cpp")
