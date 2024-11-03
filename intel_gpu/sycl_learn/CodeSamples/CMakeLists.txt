cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

project(SyclSamples LANGUAGES CXX)

set(Utils  ${CMAKE_SOURCE_DIR}/utils)

add_subdirectory(01_HelloSycl)
add_subdirectory(02_sycl_ocl_interoperate)
