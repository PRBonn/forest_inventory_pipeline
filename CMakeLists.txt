# This project is intended to be used purely with Python as of now. The Python
# build system uses scikit-build-core which handles installing the CMake
# dependency when using build isolation.
cmake_minimum_required(VERSION 3.28)
project(
  forest_inventory_pipeline
  VERSION 0.0.1
  LANGUAGES CXX)

add_subdirectory(cpp)
