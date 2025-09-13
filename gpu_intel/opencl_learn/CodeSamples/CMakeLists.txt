cmake_minimum_required(VERSION 3.20)

project(OpenCLSamples LANGUAGES C CXX)

# 设置 C++ 标准为 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Utils  ${CMAKE_SOURCE_DIR}/utils)

add_subdirectory(01_HelloOpenCL)
add_subdirectory(02_Rope_kernel_opencl)
add_subdirectory(03_DPP_algo)
add_subdirectory(04_array_max)
add_subdirectory(05_GEMM)