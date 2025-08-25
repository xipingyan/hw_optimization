cmake_minimum_required(VERSION 3.20)

project(OpenCLSamples LANGUAGES C CXX)

set(Utils  ${CMAKE_SOURCE_DIR}/utils)

add_subdirectory(01_HelloOpenCL)
add_subdirectory(02_Rope_kernel_opencl)
add_subdirectory(03_DPP_algo)
add_subdirectory(04_array_max)

